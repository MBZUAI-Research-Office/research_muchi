#!/Users/xiangruike/miniconda3/envs/dbrx_poc/bin/python

from collections.abc import AsyncGenerator
from concurrent import futures
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import argparse
import asyncio
import inspect
import json
import logging
import pickle
import time

import grpc
import shard_pb2
import shard_pb2_grpc

import mlx.core as mx
import mlx.nn as nn

from transformers import AutoTokenizer, PreTrainedTokenizer

from serialization_utils import mx_to_bytes, bytes_to_mx

DEFAULT_TEMP = 0.6
DEFAULT_SEED = 7

# coroutines to be invoked when the event loop is shutting down
# copied from:
# https://github.com/grpc/grpc/blob/master/examples/python/helloworld/async_greeter_server_with_graceful_shutdown.py
_cleanup_coroutines = []


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


class MoeShard:

    def __init__(
        self,
        url: str,
        experts: dict,
    ) -> None:
        self.url = url
        self.other_shards = None  # set when inference call is made
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
        await shard.Receive(
            shard_pb2.ShardOuts(url=self.url, data=arr_bytes, arr_map=arr_map_bytes)
        )

    async def __call__(self, inputs: mx.array, jobs: list) -> None:
        # sample jobs:
        # [[{14}, 1], [{}, 2]]
        # for each job,
        # job[0] indicates activated experts in this shard for inputs[i]
        # job[1] indicates num additional calculations needed to avoid
        # wire memory driver activity from surfacing

        def mlp(x, v1, w1, w2):
            y = (self.act_fn(x @ w1) * (x @ v1)) @ w2
            mx.eval(y)
            return y

        expert_outs = []
        arr_map = {}

        for e in self.experts:
            v1, w1, w2 = next(self.experts[e]["generator"])
            for i, x in enumerate(inputs):
                if e in jobs[i][0]:
                    expert_outs.append(mlp(x, v1, w1, w2))
                    arr_map[f"{i}.{e}"] = len(expert_outs) - 1
                elif jobs[i][1] > 0:
                    mlp(x, v1, w1, w2)
                    jobs[i][1] -= 1

        # bc cannot serialize an empty array
        if len(expert_outs) == 0:
            expert_outs.append(mx.array([False]))

        expert_outs = mx.stack(expert_outs, axis=0)
        mx.eval(expert_outs)

        arr_bytes = mx_to_bytes(expert_outs)
        arr_map_bytes = pickle.dumps(arr_map)

        async with asyncio.TaskGroup() as tg:
            for shard in self.other_shards:
                tg.create_task(self.send(shard, arr_bytes, arr_map_bytes))

        return expert_outs, arr_map


class DistributedMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.d_model = args.d_model
        self.num_experts_per_tok = args.ffn_config["moe_top_k"]
        self.expert_map = args.ffn_config["expert_map"]
        self.router = Router(args.d_model, args.ffn_config["moe_num_experts"])

    def design_jobs(self, inds: list[list[int]], my_url: str) -> list:
        jobs = []

        for activated_experts in inds:
            job = [set(), 0]
            shard_loads = {}

            for e in activated_experts:
                url = self.expert_map[e]
                if url == my_url:
                    job[0].add(e)
                shard_loads[url] = shard_loads.get(url, 0) + 1

            job[1] = max(shard_loads.values()) - len(job[0])
            jobs.append(job)

        return jobs

    async def __call__(
        self, x: mx.array, shard: MoeShard, buffer: dict, sync_complete: asyncio.Event
    ) -> mx.array:
        ne = self.num_experts_per_tok
        orig_shape = x.shape
        x = x.reshape(-1, x.shape[-1])

        gates = self.router(x)
        gates = mx.softmax(gates.astype(mx.float32), axis=-1)

        inds = mx.stop_gradient(mx.argpartition(-gates, kth=ne, axis=-1)[:, :ne])
        scores = mx.take_along_axis(gates, inds, axis=-1)
        scores = scores / mx.linalg.norm(scores, ord=1, axis=-1, keepdims=True)
        scores = scores.astype(x.dtype)
        mx.eval(inds, scores)

        inds = inds.tolist()
        jobs = self.design_jobs(inds, shard.url)

        expert_outs, arr_map = await shard(x, jobs)
        await sync_complete.wait()
        # here bc other shards could have filled the buffer before this shard finishes
        buffer[shard.url] = {"expert_outs": expert_outs, "arr_map": arr_map}

        y = []

        for bi, st, it in zip(range(x.shape[0]), scores, inds):
            yt = []
            for e in it:
                url = self.expert_map[e]
                expert_outs = buffer[url]["expert_outs"]
                eoi = buffer[url]["arr_map"][f"{bi}.{e}"]
                yt.append(expert_outs[eoi])

            yt = mx.stack(yt, axis=-1)
            yt = (yt * st).sum(axis=-1)
            y.append(yt)

        y = mx.stack(y, axis=0)
        return y.reshape(orig_shape)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.ffn = DistributedMoeBlock(args)
        self.norm_attn_norm = NormAttnNorm(args)

    async def __call__(
        self,
        x: mx.array,
        shard: MoeShard,
        buffer: dict,
        sync_complete: asyncio.Event,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        r, h, cache = self.norm_attn_norm(x, mask, cache)
        out = (await self.ffn(h, shard, buffer, sync_complete)) + r
        return out, cache


class DBRX(nn.Module):
    def __init__(self, args: ModelArgs, experts: mx.array):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.wte = nn.Embedding(args.vocab_size, args.d_model)
        self.blocks = [DecoderLayer(args) for _ in range(args.n_layers)]
        self.moe_shard = MoeShard(args.ffn_config["shard_url"], experts)
        self.buffer = {}
        self.sync_complete = asyncio.Event()
        self.norm_f = nn.LayerNorm(args.d_model, bias=False)
        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)

    def reset_buffer_mechanism(self):
        self.buffer = {}
        self.sync_complete.clear()

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

        self.moe_shard.reset_expert_generators()
        for e, layer in enumerate(self.blocks):
            h, cache[e] = await layer(
                h, self.moe_shard, self.buffer, self.sync_complete, mask, cache[e]
            )
            self.reset_buffer_mechanism()

        return self.lm_head(self.norm_f(h)), cache


class ShardServicer(shard_pb2_grpc.ShardServicer):

    def __init__(self, model_path: str, config_filename: str) -> None:
        mx.random.seed(DEFAULT_SEED)
        self.model_path = Path(model_path)
        self.model_args = self.get_model_args(config_filename)
        self.model = self.load_model()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.other_shards = None

    def get_model_args(self, config_filename: str) -> ModelArgs:
        try:
            with open(self.model_path / config_filename, "r") as f:
                config = json.load(f)
        except FileNotFoundError:
            logging.error(f"{config_filename} not found in {self.model_path}")
            raise

        model_args = ModelArgs.from_dict(config)
        model_args.ffn_config["expert_map"] = {}

        for url, assigned_experts in model_args.ffn_config["shard_map"].items():
            for e in assigned_experts:
                model_args.ffn_config["expert_map"][e] = url

        return model_args

    def load_model(self) -> DBRX:
        # sample:
        # {0: {"weights": mx.array([0, 1, 2, 3])}}
        experts = {
            e: mx.load(str(self.model_path / f"expert{e}.safetensors"))
            for e in self.model_args.ffn_config["assigned_experts"]
        }
        mx.eval(experts)

        model = DBRX(self.model_args, experts)
        non_expert_weights = mx.load(str(self.model_path / f"non-expert.safetensors"))
        model.load_weights(list(non_expert_weights.items()))
        mx.eval(model.parameters())
        model.eval()

        return model

    def reset(self):
        self.other_shards = None

    async def generate_step(
        self,
        model: nn.Module,
        prompt: mx.array,
        temp: float,
    ) -> AsyncGenerator[Tuple[mx.array, mx.array]]:

        def sample(logits: mx.array) -> Tuple[mx.array, float]:
            softmax_logits = mx.softmax(logits)

            if temp == 0:
                token = mx.argmax(logits, axis=-1)
            else:
                token = mx.random.categorical(logits * (1 / temp))

            prob = softmax_logits[0, token]
            return token, prob

        y = prompt
        cache = None

        while True:
            logits, cache = await model(y[None], cache=cache)
            logits = logits[:, -1, :]
            y, prob = sample(logits)
            yield y, prob

    async def generate(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        prompt: str,
        max_tokens: int,
        temp: float,
    ) -> shard_pb2.Outputs:
        prompt_tokens = mx.array(tokenizer.encode(prompt))

        tic = time.perf_counter()
        tokens = []
        token_strings = []
        REPLACEMENT_CHAR = "\ufffd"

        n = 0
        async for token, prob in self.generate_step(
            model,
            prompt_tokens,
            temp,
        ):
            if n >= max_tokens:
                break

            token = token.item()  # get word ID
            if n == 0:
                prompt_time = time.perf_counter() - tic
                tic = time.perf_counter()
            if token == tokenizer.eos_token_id:
                n += 1
                break
            tokens.append(token)

            s = tokenizer.decode(tokens)  # str
            # Reset token cache at line break
            if s[-1] == "\n":
                tokens = []
                token_strings.append(s)

            n += 1

        token_strings.append(tokenizer.decode(tokens).replace(REPLACEMENT_CHAR, ""))
        gen_time = time.perf_counter() - tic

        return shard_pb2.Outputs(
            prompt_time=prompt_time,
            prompt_t_cnt=prompt_tokens.size,
            gen_time=gen_time,
            gen_t_cnt=n - 1,
            response="".join(token_strings),
        )

    async def Start(self, request: shard_pb2.Inputs, context) -> None:
        async with AsyncExitStack() as es:
            self.other_shards = []

            for url in self.model_args.ffn_config["shard_map"]:
                if url == self.model_args.ffn_config["shard_url"]:
                    continue
                channel = await es.enter_async_context(grpc.aio.insecure_channel(url))
                shard = shard_pb2_grpc.ShardStub(channel)
                self.other_shards.append(shard)

            self.model.moe_shard.other_shards = self.other_shards
            response = await self.generate(
                self.model,
                self.tokenizer,
                request.prompt,
                request.max_tokens,
                DEFAULT_TEMP,
            )

        return shard_pb2.Outputs(response=response)

    def Receive(self, request: shard_pb2.ShardOuts, context):
        self.model.buffer[request.url] = {
            "expert_outs": bytes_to_mx(request.data),
            "arr_map": pickle.loads(request.arr_map),
        }

        if len(self.model.buffer) == len(self.other_shards):
            self.model.sync_complete.set()

        return shard_pb2.Empty()


async def serve(port: int, model_path: str, config_filename: str):
    server = grpc.aio.server()
    shard_pb2_grpc.add_ShardServicer_to_server(
        ShardServicer(model_path, config_filename), server
    )
    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)
    await server.start()
    logging.info(f"server started, listening on {listen_addr}")

    # copied from:
    # https://github.com/grpc/grpc/blob/master/examples/python/helloworld/async_greeter_server_with_graceful_shutdown.py
    async def server_graceful_shutdown():
        logging.info("Starting graceful shutdown...")
        # Shuts down the server with 3 seconds of grace period. During the
        # grace period, the server won't accept new connections and allow
        # existing RPCs to continue within the grace period.
        await server.stop(3)

    _cleanup_coroutines.append(server_graceful_shutdown())
    await server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int)
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--config-filename", type=str)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(serve(args.port, args.model_path, args.config_filename))
    finally:
        loop.run_until_complete(*_cleanup_coroutines)
        loop.close()
