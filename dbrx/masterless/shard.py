#!/Users/xiangruike/miniconda3/envs/dbrx_poc/bin/python

from concurrent import futures
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import argparse
import asyncio
import inspect
import json
import logging
import pickle

import grpc
import shard_pb2
import shard_pb2_grpc

import mlx.core as mx
import mlx.nn as nn

from serialization_utils import mx_to_bytes, bytes_to_mx


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


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.ffn = DistributedSparseMoeBlock(args)
        self.norm_attn_norm = NormAttnNorm(args)

    async def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        r, h, cache = self.norm_attn_norm(x, mask, cache)
        out = (await self.ffn(h)) + r
        return out, cache


class DBRX(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.wte = nn.Embedding(args.vocab_size, args.d_model)
        self.blocks = [DecoderLayer(args) for _ in range(args.n_layers)]
        self.moe = MoE()
        self.norm_f = nn.LayerNorm(args.d_model, bias=False)
        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)

    async def __call__(
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

        self.moe.reset_expert_generators()
        for e, layer in enumerate(self.blocks):
            h, cache[e] = await layer(h, mask, cache[e])

        return self.lm_head(self.norm_f(h)), cache


class MoeShard:

    def __init__(
        self, other_shards: list[shard_pb2_grpc.ShardStub], experts: dict
    ) -> None:
        self.other_shards = other_shards
        self.experts = experts
        self.act_fn = nn.silu

    def get_expert_generator(self, e: int):
        v1, w1 = None, None
        for i, weight in enumerate(self.experts[e]["weights"]):
            if i % 3 == 0:
                v1 = weight.T
            elif i % 3 == 1:
                w1 = weight.T
            else:
                yield v1, w1, weight

    def reset_expert_generators(self):
        for e in self.experts:
            self.experts[e]["generator"] = self.get_expert_generator(e)

    async def send(
        self, shard: shard_pb2_grpc.ShardStub, arr_bytes: bytes, arr_map_bytes: bytes
    ):
        await shard.Receive(shard_pb2.ExpertOuts(data=arr_bytes, arr_map=arr_map_bytes))

    async def __call__(self, inputs: mx.array, jobs: list) -> Tuple[mx.array, dict]:
        # sample jobs:
        # [[{14}, 1], [{}, 2]]
        # for each job,
        # job[0] indicates activated experts in this shard for inputs[i]
        # job[1] indicates num additional calculations needed to avoid
        # wire memory driver activity from surfacing

        expert_outs = []
        arr_map = {}

        for e in self.experts:
            v1, w1, w2 = next(self.experts[e]["generator"])
            for i, x in enumerate(inputs):
                if e in jobs[i][0]:
                    y = (self.act_fn(x @ w1) * (x @ v1)) @ w2
                    mx.eval(y)
                    expert_outs.append(y)
                    arr_map[f"{i}.{e}"] = len(expert_outs) - 1
                elif jobs[i][1] > 0:
                    mx.eval((self.act_fn(x @ w1) * (x @ v1)) @ w2)
                    jobs[i][1] -= 1

        arr_bytes = mx_to_bytes(mx.stack(expert_outs, axis=0))
        arr_map_bytes = pickle.dumps(arr_map)

        async with asyncio.TaskGroup() as tg:
            for shard in self.other_shards:
                tg.create_task(self.send(shard, arr_bytes, arr_map_bytes))


class DistributedMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.d_model = args.d_model
        self.num_experts_per_tok = args.ffn_config["moe_top_k"]
        self.shard_url = args.ffn_config["shard_url"]
        self.shard_map = args.ffn_config["shard_map"]
        self.expert_map = args.ffn_config["expert_map"]
        self.router = Router(args.d_model, args.ffn_config["moe_num_experts"])
        # self.activated_experts = None
        # self.expert_outs = None

    def reset(self) -> None:
        pass

    def receive(self, shard_url: str, expert_outs: mx.array) -> None:
        pass

    def design_jobs(self, inds: list[list[int]]):
        jobs = []

        for activated_experts in inds:
            job = [set(), 0]
            shard_loads = {}

            for e in activated_experts:
                url = self.expert_map[e]
                if url == self.shard_url:
                    job[0].add(e)
                shard_loads[url] = shard_loads.get(url, 0) + 1

            job[1] = max(shard_loads.values()) - len(job[0])
            jobs.append(job)

        return jobs

    async def __call__(self, x: mx.array) -> mx.array:
        ne = self.num_experts_per_tok
        orig_shape = x.shape
        x = x.reshape(-1, x.shape[-1])

        gates = self.router(x)
        gates = mx.softmax(gates.astype(mx.float32), axis=-1)

        inds = mx.stop_gradient(mx.argpartition(-gates, kth=ne, axis=-1)[:, :ne])
        scores = mx.take_along_axis(gates, inds, axis=-1)
        scores = scores / mx.linalg.norm(scores, ord=1, axis=-1, keepdims=True)
        scores = scores.astype(x.dtype)

        # mx.eval(inds, scores)

        y = []
        batch_size = x.shape[0]

        for xt, st, it in zip(x, scores, inds.tolist()):
            pass

        y = mx.stack(y, axis=0)
        return y.reshape(orig_shape)


class ShardServicer(shard_pb2_grpc.ShardServicer):

    def __init__(self, model_path: str, config_filename: str) -> None:
        self.model_path = Path(model_path)
        self.model_args = self.get_model_args(config_filename)
        self.model = self.load_model()

    def Receive(self, request: shard_pb2.ExpertOuts, context):
        bytes_to_mx(request.data)
        return shard_pb2.Empty()

    def get_model_args(self, config_filename: str) -> dict:
        try:
            with open(self.model_path / config_filename, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logging.error(f"{config_filename} not found in {self.model_path}")
            raise

    def load_model(self):
        # sample:
        # {0: {"weights": mx.array([0, 1, 2, 3])}}
        experts = {
            e: mx.load(str(self.model_path / f"expert{e}.safetensors"))
            for e in self.model_args["ffn_config"]["assigned_experts"]
        }
        mx.eval(experts)

        return DistributedDBRX(experts)


def serve(port: int, model_path: str, config_filename: str):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    shard_pb2_grpc.add_ShardServicer_to_server(
        ShardServicer(model_path, config_filename), server
    )
    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)
    server.start()
    logging.info(f"server started, listening on {listen_addr}")
    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int)
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--config-filename", type=str)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    serve(args.port, args.model_path, args.config_filename)
