#!/Users/xiangruike/miniconda3/envs/dbrx_poc/bin/python

from collections import OrderedDict
from dataclasses import dataclass
from multiprocessing import connection
from pathlib import Path
from typing import Any, Optional, Tuple
import concurrent.futures
import inspect
import json
import logging
import pickle
import time


import mlx.core as mx
import mlx.nn as nn

from transformers import AutoTokenizer

from serialization_utils import mx_to_bytes, bytes_to_mx

DEFAULT_TEMP = 0.6
DEFAULT_SEED = 7

LOGS = {
    "moe_lat": [],
    "comm_lat": [],
    "experts_act": [],
}


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

    def __init__(self, experts: dict) -> None:
        self.experts = experts
        self.setup_generators()
        self.expert_lru = LruCache.fromkeys(experts.keys())

    def setup_generators(self):

        def expert_generator(e):
            v1, w1 = None, None
            while True:
                for i, weight in enumerate(self.experts[e]["weights"]):
                    if i % 3 == 0:
                        v1 = weight
                    elif i % 3 == 1:
                        w1 = weight
                    else:
                        yield v1, w1, weight

        for e in self.experts:
            self.experts[e]["generator"] = expert_generator(e)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_heads = args.n_heads
        self.d_model = args.d_model
        self.head_dim = args.d_model // args.n_heads
        self.num_key_value_heads = args.attn_config["kv_n_heads"]
        self.clip_qkv = args.attn_config["clip_qkv"]
        self.rope_theta = args.attn_config["rope_theta"]

        self.scale = self.head_dim**-0.5

        self.Wqkv = nn.Linear(
            args.d_model,
            (self.num_key_value_heads * 2 + self.num_heads) * self.head_dim,
            bias=False,
        )
        self.out_proj = nn.Linear(args.d_model, args.d_model, bias=False)
        self.rope = nn.RoPE(
            self.head_dim,
            traditional=False,
            base=self.rope_theta,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:

        qkv = self.Wqkv(x)
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
        return self.out_proj(output), (keys, values)


class NormAttnNorm(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.norm_1 = nn.LayerNorm(args.d_model, bias=False)
        self.norm_2 = nn.LayerNorm(args.d_model, bias=False)
        self.attn = Attention(args)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        h, cache = self.attn(self.norm_1(x), mask=mask, cache=cache)
        x = h + x
        return x, self.norm_2(x), cache


class Router(nn.Module):
    def __init__(self, d_model: int, num_experts: int):
        super().__init__()
        self.layer = nn.Linear(d_model, num_experts, bias=False)

    def __call__(self, x: mx.array):
        return self.layer(x)


class DistributedMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_num: int):
        super().__init__()
        self.act_fn = nn.silu
        self.layer_num = layer_num
        self.d_model = args.d_model
        self.ne = args.ffn_config["moe_top_k"]

        self.url = args.ffn_config["shard_url"]
        self.e_to_g = args.ffn_config["e_to_g"]
        self.dlb_groups = args.ffn_config["dlb_groups"]
        self.n_oth_shards = sum(len(d["members"]) for d in self.dlb_groups.values()) - 1
        self.router = Router(args.d_model, args.ffn_config["moe_num_experts"])

    def allocate_jobs(
        self, inds: list[list[int]], expert_lru: LruCache, dry_run: bool
    ) -> tuple:
        jobs = []
        job_map = []
        max_loads = []

        for activated_experts in inds:
            by_dlb_group = {}
            by_shard = {}
            jm = {}

            for e in activated_experts:
                by_dlb_group.setdefault(self.e_to_g[e], []).append(e)

            for g, es in by_dlb_group.items():
                members = self.dlb_groups[g]["members"]
                for i, e in enumerate(es):
                    url = members[i % len(members)]
                    by_shard.setdefault(url, set()).add(e)
                    jm[e] = url

            jobs.append(by_shard.get(self.url, set()))
            job_map.append(jm)
            max_loads.append(max(len(v) for v in by_shard.values()))

        for i in range(len(jobs)):
            for e in jobs[i]:
                expert_lru.move_to_end(e)

            n_warmups = max_loads[i] - len(jobs[i])
            for _ in range(n_warmups):
                jobs[i].add(expert_lru.get_lru())

            if not dry_run:
                LOGS["experts_act"].append(len(jobs[i]))

        return jobs, job_map

    def moe_shard(self, x: mx.array, jobs: list[set], weights: RawWeights):
        ws = {e: next(d["generator"]) for e, d in weights.experts.items()}
        for xt, job in zip(x, jobs):
            expert_outs = []
            arr_map = {}
            for e in job:
                v1, w1, w2 = ws[e]
                expert_outs.append((self.act_fn(xt @ w1.T) * (xt @ v1.T)) @ w2)
                arr_map[e] = len(expert_outs) - 1

            expert_outs = mx.stack(expert_outs, axis=0)
            yield expert_outs, arr_map

    def all_dispatch(
        self,
        x: mx.array,
        jobs: list[set],
        weights: RawWeights,
        send_conn: connection.Connection,
    ):
        tic = time.perf_counter_ns()

        shard_outs = {}
        for bi, (expert_outs, arr_map) in enumerate(self.moe_shard(x, jobs, weights)):
            mx.eval(expert_outs)
            shard_outs[bi] = (expert_outs, arr_map)
            send_conn.send_bytes(mx_to_bytes(expert_outs))
            send_conn.send_bytes(pickle.dumps((self.url, self.layer_num, bi, arr_map)))

        return {self.url: shard_outs}, time.perf_counter_ns() - tic

    def all_combine(
        self,
        batch_size: int,
        resv_conn: connection.Connection,
    ):
        shard_outs = {}
        for _ in range(batch_size * self.n_oth_shards):
            expert_outs = bytes_to_mx(resv_conn.recv_bytes())
            url, li, bi, arr_map = pickle.loads(resv_conn.recv_bytes())
            shard_outs.setdefault(url, {})[bi] = (expert_outs, arr_map)
        return shard_outs

    def __call__(
        self,
        x: mx.array,
        raw_weights: RawWeights,
        resv_conn: connection.Connection,
        send_conn: connection.Connection,
        executor: concurrent.futures.ThreadPoolExecutor,
        dry_run: bool,
    ) -> mx.array:
        orig_shape = x.shape  # (sample_size, sequence_length, d_model)
        x = x.reshape(-1, x.shape[-1])  # (sample_size * sequence_length, d_model)

        gates = self.router(x)
        gates = mx.softmax(gates.astype(mx.float32), axis=-1)

        inds = mx.stop_gradient(
            mx.argpartition(-gates, kth=self.ne, axis=-1)[:, : self.ne]
        )
        scores = mx.take_along_axis(gates, inds, axis=-1)
        scores = scores / mx.linalg.norm(scores, ord=1, axis=-1, keepdims=True)
        scores = scores.astype(x.dtype)
        mx.eval(inds, scores)

        inds = inds.tolist()
        jobs, job_map = self.allocate_jobs(inds, raw_weights.expert_lru, dry_run)
        batch_size = x.shape[0]

        tic = time.perf_counter_ns()

        comm_fut = executor.submit(self.all_combine, batch_size, resv_conn)
        shard_outs, moe_lat = self.all_dispatch(x, jobs, raw_weights, send_conn)
        shard_outs.update(comm_fut.result())

        if not dry_run:
            LOGS["moe_lat"].append(moe_lat)
            LOGS["comm_lat"].append(time.perf_counter_ns() - tic - moe_lat)

        y = []

        for bi, st, it in zip(range(batch_size), scores, inds):
            yt = []
            for e in it:
                url = job_map[bi][e]
                expert_outs, arr_map = shard_outs[url][bi]
                yt.append(expert_outs[arr_map[e]])

            yt = mx.stack(yt, axis=-1)
            yt = (yt * st).sum(axis=-1)
            y.append(yt)

        y = mx.stack(y, axis=0)
        return y.reshape(orig_shape)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_num: int):
        super().__init__()
        self.ffn = DistributedMoeBlock(args, layer_num)
        self.norm_attn_norm = NormAttnNorm(args)

    def __call__(
        self,
        x: mx.array,
        raw_weights: RawWeights,
        dry_run: bool,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ):
        r, h, cache = self.norm_attn_norm(x, mask, cache)
        return h + r, cache
        # out = self.ffn(h, raw_weights, resv_conn, send_conn, executor, dry_run) + r
        # return out, cache


class DBRX(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        raw_weights: RawWeights,
    ):
        super().__init__()
        self.n_layers = args.n_layers
        self.raw_weights = raw_weights
        self.wte = nn.Embedding(args.vocab_size, args.d_model)
        self.blocks = [DecoderLayer(args, i) for i in range(args.n_layers)]
        self.norm_f = nn.LayerNorm(args.d_model, bias=False)
        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)

    def out_transform(self, x: mx.array, temp: float) -> mx.array:
        logits = self.lm_head(self.norm_f(x))
        logits = logits[:, -1, :]
        if temp == 0:
            return mx.argmax(logits, axis=-1)
        return mx.random.categorical(logits * (1 / temp))

    def prewarm(self):
        for li in range(self.n_layers):
            xs = []
            for d in self.raw_weights.experts.values():
                xs.extend(next(d["generator"]))
            y = mx.sum(mx.stack(xs, axis=0), axis=0)
            mx.eval(y)

    def __call__(
        self,
        inputs: mx.array,
        temp: float,
        cache: Optional[list[Tuple[mx.array, mx.array]]] = None,
        dry_run: Optional[bool] = False,
    ):
        print(inputs)
        h = self.wte(inputs)

        mask = None
        T = h.shape[1]
        if T > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
            mask = mask.astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.blocks)

        h, cache = self.blocks[0](
            h,
            self.raw_weights,
            dry_run,
            mask=mask,
            cache=cache[0],
        )
        h = self.out_transform(h, temp)
        tokens = [[ht.item()] for ht in h]
        return tokens
        # print(h.item())
        # for id in h:
        #     print(id.item(), type(id.item()))

        # for e, layer in enumerate(self.blocks):
        #     h, updated_cache = layer(
        #         h,
        #         self.raw_weights,
        #         dry_run,
        #         mask=mask,
        #         cache=cache[e],
        #     )
        #     if dry_run:
        #         continue
        #     cache[e] = updated_cache

        # out = self.out_transform(h, temp)
        # mx.eval(out)
        # return out, cache


class Test:

    def __init__(self, model_path: str, config_filename: str) -> None:
        mx.random.seed(DEFAULT_SEED)
        self.model_path = Path(model_path)
        self.model_args = self.get_model_args(config_filename)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.model = self.load_model()

    def get_model_args(self, config_filename: str) -> ModelArgs:
        try:
            with open(self.model_path / config_filename, "r") as f:
                config = json.load(f)
        except FileNotFoundError:
            logging.error(f"{config_filename} not found in {self.model_path}")
            raise

        model_args = ModelArgs.from_dict(config)
        model_args.ffn_config["e_to_g"] = {}

        for g, d in model_args.ffn_config["dlb_groups"].items():
            for e in d["experts"]:
                model_args.ffn_config["e_to_g"][e] = g

            if model_args.ffn_config["shard_url"] in d["members"]:
                model_args.ffn_config["assigned_experts"] = d["experts"]

        return model_args

    def load_model(self) -> DBRX:
        non_experts = mx.load(str(self.model_path / f"non-expert.safetensors"))
        # sample:
        # {0: {"weights": mx.array([0, 1, 2, 3])}}
        # experts = {
        #     e: mx.load(str(self.model_path / f"expert{e}.safetensors"))
        #     for e in self.model_args.ffn_config["assigned_experts"]
        # }
        # mx.eval(experts)
        experts = {}

        raw_weights = RawWeights(experts)
        model = DBRX(self.model_args, raw_weights)
        model.load_weights(list(non_experts.items()))
        mx.eval(model.parameters())
        model.eval()

        return model

    def generate(self, prompts: list[str], max_tokens: int, temp: float):
        prompt_tokens = mx.array(self.tokenizer(prompts)["input_ids"])
        y = prompt_tokens
        cache = None
        tokens = []
        token_strings = []
        REPLACEMENT_CHAR = "\ufffd"

        tokens = self.model(y, temp, cache=cache)
        print(tokens)
        print(self.tokenizer.batch_decode(tokens))
        # self.model.prewarm()

        # for n in range(max_tokens):
        #     y, cache = self.model(y, temp, cache=cache)

        #     token = y.item()  # get word ID
        #     if n == 0:
        #         if token != self.tokenizer.eos_token_id:
        #             self.send_conn.send(True)
        #             # token generation dry run
        #             self.model(y[None], temp, executor, cache=cache, dry_run=True)
        #         else:
        #             self.send_conn.send(False)  # signal no dry run
        #         prompt_time = time.perf_counter() - tic
        #         tic = time.perf_counter()
        #     if token == self.tokenizer.eos_token_id:
        #         self.send_conn.send(False)
        #         break
        #     tokens.append(token)

        #     s = self.tokenizer.decode(tokens)  # str
        #     # Reset token cache at line break
        #     if s[-1] == "\n":
        #         tokens = []
        #         token_strings.append(s)

        #     self.send_conn.send(True)

        # token_strings.append(
        #     self.tokenizer.decode(tokens).replace(REPLACEMENT_CHAR, "")
        # )
        # gen_time = time.perf_counter() - tic

        # return [
        #     prompt_time,
        #     prompt_tokens.size,
        #     gen_time,
        #     n,
        #     "".join(token_strings),
        # ]


if __name__ == "__main__":
    test = Test(
        "/Users/xiangruike/dbrx-instruct/distributable/batch2",
        "v4.2_2n_shard_config.json",
    )
    test.generate(["hello",  "there"], 10, 0.1)
