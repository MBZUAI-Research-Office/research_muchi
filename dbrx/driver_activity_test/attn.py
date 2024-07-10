#!/Users/xiangruike/miniconda3/envs/dbrx_poc/bin/python

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
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


class RawWeights:

    def __init__(self, n_layers: int, non_experts: dict) -> None:
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

        self.raw_ptrs = raw_ptrs
        self.lib_ptrs = lib_ptrs
        self.ne_warmup = ne_warmup

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


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_num: int):
        super().__init__()
        self.layer_num = layer_num
        self.norm_attn_norm = NormAttnNorm(args, layer_num)

    def __call__(
        self,
        x: mx.array,
        raw_weights: RawWeights,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        r, h, cache = self.norm_attn_norm(x, raw_weights, mask, cache)

        orig_shape = h.shape  # (sample_size, sequence_length, d_model)
        h = h.reshape(-1, h.shape[-1])  # (sample_size * sequence_length, d_model)
        ws = raw_weights(self.layer_num)

        gates = h @ ws["router"].T
        gates = mx.softmax(gates.astype(mx.float32), axis=-1)

        inds = mx.stop_gradient(mx.argpartition(-gates, kth=4, axis=-1)[:, :4])
        scores = mx.take_along_axis(gates, inds, axis=-1)
        scores = scores / mx.linalg.norm(scores, ord=1, axis=-1, keepdims=True)
        scores = scores.astype(h.dtype)
        mx.eval(inds, scores)

        return h + r, cache


class DBRX(nn.Module):
    def __init__(self, args: ModelArgs, raw_weights: RawWeights):
        super().__init__()
        self.n_layers = args.n_layers
        self.raw_weights = raw_weights
        self.wte = nn.Embedding(args.vocab_size, args.d_model)
        self.blocks = [DecoderLayer(args, i) for i in range(args.n_layers)]
        self.norm_f = nn.LayerNorm(args.d_model, bias=False)

    def prewarm(self):
        tic = time.perf_counter_ns()

        vecs = self.raw_weights.ne_warmup
        for _ in range(self.n_layers):
            mx.eval(mx.sum(mx.stack(vecs, axis=0), axis=0))

        print(
            f"avg prewarm time: {(time.perf_counter_ns() - tic) / self.n_layers / 1000**2} ms",
            flush=True,
        )

    def __call__(self, inputs: mx.array, cache=None):
        h = self.wte(inputs)

        mask = None
        T = h.shape[1]
        if T > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
            mask = mask.astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.blocks)

        for e, layer in enumerate(self.blocks):
            h, cache[e] = layer(h, self.raw_weights, mask, cache[e])

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
        mx.eval(non_experts)

        raw_weights = RawWeights(self.model_args.n_layers, non_experts)
        model = DBRX(self.model_args, raw_weights)
        model.load_weights(list(raw_weights.lib_ptrs.items()))
        model.eval()

        return model

    def start(self):
        # shows that not warming norm_1 & norm_2 causes driver processing
        x = mx.array([[1]], dtype=mx.int32)
        mx.eval(x)
        self.model.prewarm()
        for _ in range(3):
            self.model(x)


if __name__ == "__main__":
    test = Test(
        "/Users/xiangruike/dbrx-instruct/distributable/batch2",
        "v4.2_2n_shard_config.json",
    )
    test.start()
