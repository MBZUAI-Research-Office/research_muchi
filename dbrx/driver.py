#!/Users/xiangruike/miniconda3/envs/dbrx_poc/bin/python

from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Optional, Tuple
import argparse
import asyncio
import logging
import time

import grpc
import moe_shard_pb2
import moe_shard_pb2_grpc

import numpy as np

import mlx.core as mx
import mlx.nn as nn

from base import BaseModelArgs

NUM_LAYERS = 40
EMBEDDING_LENGTH = 6144
NUM_EXPERTS = 4
TOP_K = 4
DUMMY_NP_DATA = np.random.uniform(-1, 1, EMBEDDING_LENGTH).astype(np.float32)
# EXPERT_CHANNELS = [
#     "169.254.238.2:2000",
#     "169.254.238.4:4000",
#     "169.254.238.5:5000",
#     "169.254.238.6:6000",
# ]
EXPERT_CHANNELS = [
    # "169.254.136.2:2000",
    # "169.254.136.4:4000",
    # "169.254.136.5:5000",
    "169.254.136.6:6000",
]


@dataclass
class ModelArgs(BaseModelArgs):
    vocab_size: int
    d_model: int
    ffn_config: dict
    attn_config: dict
    n_layers: int
    n_heads: int
    moe_shards: list[moe_shard_pb2_grpc.MoeShardStub]


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


# TODO: this needs to change
class DistributedSparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.d_model = args.d_model
        self.ffn_dim = args.ffn_config["ffn_hidden_size"]
        self.num_experts = args.ffn_config["moe_num_experts"]
        self.num_experts_per_tok = args.ffn_config["moe_top_k"]

        self.router = Router(self.d_model, self.num_experts)
        self.experts = [
            MLP(self.d_model, self.ffn_dim) for _ in range(self.num_experts)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        ne = self.num_experts_per_tok
        orig_shape = x.shape
        x = x.reshape(-1, x.shape[-1])

        gates = self.router(x)
        gates = mx.softmax(gates.astype(mx.float32), axis=-1)

        inds = mx.stop_gradient(mx.argpartition(-gates, kth=ne, axis=-1)[:, :ne])
        scores = mx.take_along_axis(gates, inds, axis=-1)
        scores = scores / mx.linalg.norm(scores, ord=1, axis=-1, keepdims=True)
        scores = scores.astype(x.dtype)

        y = []
        for xt, st, it in zip(x, scores, inds.tolist()):
            yt = mx.stack([self.experts[e](xt) for e in it], axis=-1)
            yt = (yt * st).sum(axis=-1)
            y.append(yt)
        y = mx.stack(y, axis=0)

        return y.reshape(orig_shape)


class DistributedDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.ffn = DistributedSparseMoeBlock(args)
        self.norm_attn_norm = NormAttnNorm(args)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        r, h, cache = self.norm_attn_norm(x, mask, cache)
        out = self.ffn(h) + r
        return out, cache


class DistributedDBRX(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.wte = nn.Embedding(args.vocab_size, args.d_model)
        self.blocks = [DistributedDecoderLayer(args) for _ in range(args.n_layers)]
        self.norm_f = nn.LayerNorm(args.d_model, bias=False)
        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        h = self.wte(inputs)

        mask = None
        T = h.shape[1]
        if T > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
            mask = mask.astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.blocks)

        for e, layer in enumerate(self.blocks):
            h, cache[e] = layer(h, mask, cache[e])

        return self.lm_head(self.norm_f(h)), cache


async def execute_on_expert(stub, block_num, activated_experts, data, outputs):
    out = await stub.Execute(
        moe_shard_pb2.Inputs(
            block_num=block_num,
            activated_experts=activated_experts.tobytes(),
            data=data.tobytes(),
        )
    )
    outputs.append(np.frombuffer(out.data, dtype=np.float32))
    return


async def generate(num_tokens: int):
    async with AsyncExitStack() as es:
        expert_channels = [
            await es.enter_async_context(grpc.aio.insecure_channel(url))
            for url in EXPERT_CHANNELS
        ]
        experts = [
            moe_shard_pb2_grpc.MoeShardStub(channel) for channel in expert_channels
        ]
        latencies = []

        for i in range(num_tokens):
            token_latency = 0
            for j in range(NUM_LAYERS):
                # chosen_experts = np.random.randint(0, NUM_EXPERTS, size=TOP_K)
                chosen_experts = [0]  # TODO: naming change
                activated_experts = np.array([0, 1, 2, 3])
                outputs = []
                tic = time.perf_counter()

                async with asyncio.TaskGroup() as tg:
                    for k in chosen_experts:
                        tg.create_task(
                            execute_on_expert(
                                experts[k], j, activated_experts, DUMMY_NP_DATA, outputs
                            )
                        )

                token_latency += time.perf_counter() - tic
            latencies.append(token_latency)

        print("=" * 20)
        print(f"NUM LAYERS: {NUM_LAYERS}")
        print(f"EMBEDDING_LENGTH: {EMBEDDING_LENGTH}")
        print(f"TOP K: {TOP_K}")
        print(f"token latencies:\n{latencies}")
        print(f"average: {np.mean(latencies)} sec(s)")
        print("=" * 20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=1)
    args = parser.parse_args()
    logging.basicConfig()
    asyncio.run(generate(args.num_tokens))
