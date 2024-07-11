#!/Users/xiangruike/miniconda3/envs/dbrx_poc/bin/python

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple
import inspect
import json
import logging
import time

import mlx.core as mx
import mlx.nn as nn

DEFAULT_TEMP = 0.6
DEFAULT_SEED = 7


@dataclass
class ModelArgs:
    vocab_size: int
    d_model: int
    ffn_config: dict
    attn_config: dict
    n_layers: int
    n_heads: int

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


class LruCache(OrderedDict):
    # inspired by:
    # https://docs.python.org/3/library/collections.html#collections.OrderedDict
    # https://stackoverflow.com/questions/21062781/shortest-way-to-get-first-item-of-ordereddict-in-python-3

    def get_lru(self) -> Any:
        k = next(iter(self))
        self.move_to_end(k)
        return k


class RawWeights:

    def __init__(self, n_layers: int, experts: dict, non_experts: dict) -> None:
        raw_ptrs = {i: {} for i in range(n_layers)}
        lib_ptrs = {}
        for i, mat in enumerate(non_experts["wqkv_weights"]):
            raw_ptrs[i]["wqkv"] = mat
        for i, mat in enumerate(non_experts["out_proj_weights"]):
            raw_ptrs[i]["out_proj"] = mat
        for i, mat in enumerate(non_experts["router_weights"]):
            raw_ptrs[i]["router"] = mat
        for j, vec in enumerate(non_experts["norm_weights"]):
            if j == non_experts["norm_weights"].shape[0] - 1:
                lib_ptrs["norm_f.weight"] = vec
                continue
            i = j // 2
            if j % 2 == 0:
                lib_ptrs[f"blocks.{i}.norm_attn_norm.norm_1.weight"] = vec
            elif j % 2 == 1:
                lib_ptrs[f"blocks.{i}.norm_attn_norm.norm_2.weight"] = vec
        for i, mat in enumerate(non_experts["vocab_weights"]):
            if i == 0:
                lib_ptrs["wte.weight"] = mat
            elif i == 1:
                raw_ptrs["lm_head"] = mat
        for e, d in experts.items():
            for j, mat in enumerate(d["weights"]):
                i = j // 3
                if e not in raw_ptrs[i]:
                    raw_ptrs[i][e] = {}
                if j % 3 == 0:
                    raw_ptrs[i][e]["v1"] = mat
                elif j % 3 == 1:
                    raw_ptrs[i][e]["w1"] = mat
                else:
                    raw_ptrs[i][e]["w2"] = mat

        ne_warmup = []
        for vec in raw_ptrs[0]["wqkv"]:
            ne_warmup.append(vec)
            break
        for vec in raw_ptrs[0]["out_proj"]:
            ne_warmup.append(vec)
            break
        for vec in raw_ptrs[0]["router"]:
            ne_warmup.append(vec)
            break
        for vec in lib_ptrs["wte.weight"]:
            ne_warmup.append(vec)
            break
        ne_warmup.append(lib_ptrs["blocks.0.norm_attn_norm.norm_1.weight"])

        e_warmup = []
        for e in experts:
            for vec in raw_ptrs[0][e]["v1"]:
                e_warmup.append(vec)
                break

        self.raw_ptrs = raw_ptrs
        self.lib_ptrs = lib_ptrs
        self.ne_warmup = ne_warmup
        self.full_warmup = ne_warmup + e_warmup
        self.expert_lru = LruCache.fromkeys(experts.keys())

    def __call__(self, k):
        return self.raw_ptrs[k]


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_num: int):
        super().__init__()
        self.layer_num = layer_num
        self.num_heads = args.n_heads
        self.d_model = args.d_model
        self.head_dim = args.d_model // args.n_heads
        self.num_key_value_heads = args.attn_config["kv_n_heads"]
        self.clip_qkv = args.attn_config["clip_qkv"]

        self.scale = self.head_dim**-0.5
        self.rope = nn.RoPE(
            self.head_dim,
            traditional=False,
            base=args.attn_config["rope_theta"],
        )

    def __call__(
        self,
        x: mx.array,
        raw_weights: RawWeights,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        ws = raw_weights(self.layer_num)
        qkv = x @ ws["wqkv"].T
        qkv = mx.clip(qkv, a_min=-self.clip_qkv, a_max=self.clip_qkv)
        splits = [self.d_model, self.d_model + self.head_dim * self.num_key_value_heads]
        queries, keys, values = mx.split(qkv, splits, axis=-1)

        B, L, D = x.shape

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )

        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return output @ ws["out_proj"].T, (keys, values)


class NormAttnNorm(nn.Module):
    def __init__(self, args: ModelArgs, layer_num: int):
        super().__init__()
        self.norm_1 = nn.LayerNorm(args.d_model, bias=False)
        self.norm_2 = nn.LayerNorm(args.d_model, bias=False)
        self.attn = Attention(args, layer_num)

    def __call__(
        self,
        x: mx.array,
        raw_weights: RawWeights,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        h, cache = self.attn(self.norm_1(x), raw_weights, mask=mask, cache=cache)
        x = h + x
        return x, self.norm_2(x), cache


class DistributedMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_num: int):
        super().__init__()
        self.act_fn = nn.silu
        self.layer_num = layer_num
        self.d_model = args.d_model
        self.num_experts_per_tok = args.ffn_config["moe_top_k"]

        self.url = args.ffn_config["shard_url"]

    def moe_shard(self, x: mx.array, job: set, ws: dict) -> tuple[mx.array, dict]:
        expert_outs = []
        arr_map = {}
        for e in job:
            y = (self.act_fn(x @ ws[e]["w1"].T) * (x @ ws[e]["v1"].T)) @ ws[e]["w2"]
            expert_outs.append(y)
            arr_map[e] = len(expert_outs) - 1

        expert_outs = mx.stack(expert_outs, axis=0)
        return expert_outs, arr_map

    def call_shard_n_all_dispatch(
        self, x: mx.array, jobs: list[set], ws: dict, ne_warmup_vecs: list
    ):
        shard_outs = {}
        for bi, xt in enumerate(x):
            expert_outs, arr_map = self.moe_shard(xt, jobs[bi], ws)
            if len(jobs) > 1:
                ne_warmup_calc = mx.sum(mx.stack(ne_warmup_vecs, axis=0), axis=0)
                mx.eval(expert_outs, ne_warmup_calc)
            else:
                mx.eval(expert_outs)
            shard_outs.setdefault(self.url, {})[bi] = (expert_outs, arr_map)

        return shard_outs

    def __call__(self, x: mx.array, raw_weights: RawWeights) -> mx.array:
        ne = self.num_experts_per_tok
        orig_shape = x.shape  # (sample_size, sequence_length, d_model)
        x = x.reshape(-1, x.shape[-1])  # (sample_size * sequence_length, d_model)
        ws = raw_weights(self.layer_num)

        gates = x @ ws["router"].T
        gates = mx.softmax(gates.astype(mx.float32), axis=-1)

        inds = mx.stop_gradient(mx.argpartition(-gates, kth=ne, axis=-1)[:, :ne])
        scores = mx.take_along_axis(gates, inds, axis=-1)
        scores = scores / mx.linalg.norm(scores, ord=1, axis=-1, keepdims=True)
        scores = scores.astype(x.dtype)
        print("starting attn and router calculation", flush=True)
        mx.eval(inds, scores)

        print("starting moe shard calculation", flush=True)
        self.call_shard_n_all_dispatch(
            x,
            [{raw_weights.expert_lru.get_lru() for _ in range(3)}],
            ws,
            raw_weights.ne_warmup,
        )


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_num: int):
        super().__init__()
        self.ffn = DistributedMoeBlock(args, layer_num)
        self.norm_attn_norm = NormAttnNorm(args, layer_num)

    def __call__(
        self,
        x: mx.array,
        raw_weights: RawWeights,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ):
        r, h, cache = self.norm_attn_norm(x, raw_weights, mask, cache)
        self.ffn(h, raw_weights)


class DBRX(nn.Module):
    def __init__(self, args: ModelArgs, raw_weights: RawWeights):
        super().__init__()
        self.n_layers = args.n_layers
        self.raw_weights = raw_weights
        self.wte = nn.Embedding(args.vocab_size, args.d_model)
        self.blocks = [DecoderLayer(args, i) for i in range(args.n_layers)]
        self.norm_f = nn.LayerNorm(args.d_model, bias=False)

    def full_warm_calc(self) -> mx.array:
        return mx.sum(mx.stack(self.raw_weights.full_warmup, axis=0), axis=0)

    def prewarm(self):
        x = mx.ones((1, 1, 6144), dtype=mx.bfloat16)
        mx.eval(x)
        for _ in range(self.n_layers):
            mx.eval(self.full_warm_calc())
        # for layer in self.blocks:
        #     layer(x, self.raw_weights, None, None)
        print("finished prewarming", flush=True)

    def __call__(self, inputs: mx.array):
        h = self.wte(inputs)

        for layer in self.blocks:
            layer(h, self.raw_weights, None, None)

        print("starting norm_f and lm_head", flush=True)
        mx.eval(self.norm_f(h) @ self.raw_weights("lm_head").T)


class Test:

    def __init__(self, model_path: str, config_filename: str) -> None:
        mx.random.seed(DEFAULT_SEED)
        self.model_path = Path(model_path)
        self.model_args = self.get_model_args(config_filename)
        self.model = self.load_model()

    def get_model_args(self, config_filename: str) -> ModelArgs:
        try:
            with open(self.model_path / config_filename, "r") as f:
                config = json.load(f)
        except FileNotFoundError:
            logging.error(f"{config_filename} not found in {self.model_path}")
            raise

        return ModelArgs.from_dict(config)

    def load_model(self) -> DBRX:
        non_experts = mx.load(str(self.model_path / f"non-expert.safetensors"))
        # sample:
        # {0: {"weights": mx.array([0, 1, 2, 3])}}
        experts = {
            e: mx.load(str(self.model_path / f"expert{e}.safetensors"))
            for e in [0, 1, 2]
        }
        mx.eval(non_experts, experts)

        raw_weights = RawWeights(self.model_args.n_layers, experts, non_experts)
        model = DBRX(self.model_args, raw_weights)
        model.load_weights(list(raw_weights.lib_ptrs.items()))
        model.eval()

        return model

    def start(self):
        x = mx.array([[1]], dtype=mx.int32)
        mx.eval(x)
        self.model.prewarm()
        for _ in range(3):
            self.model(x)
            time.sleep(3)


if __name__ == "__main__":
    test = Test(
        "/Users/xiangruike/dbrx-instruct/distributable/batch2",
        "v4.2_2n_shard_config.json",
    )
    test.start()
