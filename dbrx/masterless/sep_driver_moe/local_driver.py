#!/Users/xiangruike/miniconda3/envs/dbrx_poc/bin/python

from collections.abc import AsyncGenerator
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
import local_driver_pb2
import local_driver_pb2_grpc
import moe_shard_pb2
import moe_shard_pb2_grpc

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
    local_driver_url: str
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


class DistributedMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.driver_url = args.local_driver_url
        self.d_model = args.d_model
        self.num_experts_per_tok = args.ffn_config["moe_top_k"]
        self.expert_map = args.ffn_config["expert_map"]
        self.router = Router(args.d_model, args.ffn_config["moe_num_experts"])

    def design_jobs(self, inds: list[list[int]]) -> list:
        jobs = []

        for activated_experts in inds:
            job = [set(), 0]
            shard_loads = {}

            for e in activated_experts:
                url = self.expert_map[e]
                if url == self.driver_url:
                    job[0].add(e)
                shard_loads[url] = shard_loads.get(url, 0) + 1

            job[1] = max(shard_loads.values()) - len(job[0])
            jobs.append(job)

        return jobs

    async def send(
        self,
        driver: local_driver_pb2_grpc.LocalDriverStub,
        layer_num: int,
        arr_bytes: bytes,
        arr_map_bytes: bytes,
    ):
        moe_shard_outs = local_driver_pb2.MoeShardOuts(
            url=self.driver_url,
            layer_num=layer_num,
            data=arr_bytes,
            arr_map=arr_map_bytes,
        )
        await driver.Receive(moe_shard_outs)

    async def __call__(
        self,
        x: mx.array,
        moe_shard: moe_shard_pb2_grpc.MoeShardStub,
        layer_num: int,
        other_drivers: list[local_driver_pb2_grpc.LocalDriverStub],
        buffer: dict,
        sync_complete: asyncio.Event,
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

        print(f"----------started pre moe_shard calculation----------", flush=True)

        mx.eval(inds, scores)

        inds = inds.tolist()
        jobs = self.design_jobs(inds)

        print(f"----------started moe_shard calculation----------", flush=True)

        moe_shard_outs = await moe_shard.Execute(
            moe_shard_pb2.Inputs(data=mx_to_bytes(x), jobs=pickle.dumps(jobs))
        )

        print(f"----------started communicating with other drivers----------", flush=True)

        async with asyncio.TaskGroup() as tg:
            for driver in other_drivers:
                tg.create_task(
                    self.send(
                        driver,
                        layer_num,
                        moe_shard_outs.data,
                        moe_shard_outs.arr_map,
                    )
                )

        print(f"----------started waiting for buffer to be filled----------", flush=True)

        await sync_complete.wait()

        print(f"----------started all-reduce calculation----------", flush=True)

        # here bc other shards could have filled the buffer before this shard finishes
        buffer[self.driver_url] = {
            "expert_outs": bytes_to_mx(moe_shard_outs.data),
            "arr_map": pickle.loads(moe_shard_outs.arr_map),
        }

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
    def __init__(self, args: ModelArgs, layer_num: int):
        super().__init__()
        self.ffn = DistributedMoeBlock(args)
        self.norm_attn_norm = NormAttnNorm(args)
        self.layer_num = layer_num
        self.buffer = {}
        self.sync_complete = asyncio.Event()

    def reset_buffer_mechanism(self):
        self.buffer = {}
        self.sync_complete.clear()

    async def __call__(
        self,
        x: mx.array,
        moe_shard: moe_shard_pb2_grpc.MoeShardStub,
        other_drivers: list[local_driver_pb2_grpc.LocalDriverStub],
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        r, h, cache = self.norm_attn_norm(x, mask, cache)
        h = await self.ffn(
            h, moe_shard, self.layer_num, other_drivers, self.buffer, self.sync_complete
        )
        return h + r, cache


class DBRX(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.wte = nn.Embedding(args.vocab_size, args.d_model)
        self.blocks = [DecoderLayer(args, i) for i in range(args.n_layers)]
        self.norm_f = nn.LayerNorm(args.d_model, bias=False)
        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)

    async def __call__(
        self,
        inputs: mx.array,
        moe_shard: moe_shard_pb2_grpc.MoeShardStub,
        other_drivers: list[local_driver_pb2_grpc.LocalDriverStub],
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
            h, cache[e] = await layer(h, moe_shard, other_drivers, mask, cache[e])
            layer.reset_buffer_mechanism()

        return self.lm_head(self.norm_f(h)), cache


class LocalDriverServicer(local_driver_pb2_grpc.LocalDriverServicer):

    def __init__(self, model_path: str, config_filename: str) -> None:
        mx.random.seed(DEFAULT_SEED)
        self.model_path = Path(model_path)
        self.model_args = self.get_model_args(config_filename)
        self.model = self.load_model()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.num_other_drivers = len(self.model_args.ffn_config["driver_map"]) - 1

    def get_model_args(self, config_filename: str) -> ModelArgs:
        try:
            with open(self.model_path / config_filename, "r") as f:
                config = json.load(f)
        except FileNotFoundError:
            logging.error(f"{config_filename} not found in {self.model_path}")
            raise

        model_args = ModelArgs.from_dict(config)
        model_args.ffn_config["expert_map"] = {}

        for url, assigned_experts in model_args.ffn_config["driver_map"].items():
            for e in assigned_experts:
                model_args.ffn_config["expert_map"][e] = url

        return model_args

    def load_model(self) -> DBRX:
        model = DBRX(self.model_args)
        non_expert_weights = mx.load(str(self.model_path / f"non-expert.safetensors"))
        model.load_weights(list(non_expert_weights.items()))
        mx.eval(model.parameters())
        model.eval()

        return model

    async def generate_step(
        self,
        model: nn.Module,
        moe_shard: moe_shard_pb2_grpc.MoeShardStub,
        other_drivers: list[local_driver_pb2_grpc.LocalDriverStub],
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
            logits, cache = await model(y[None], moe_shard, other_drivers, cache=cache)
            logits = logits[:, -1, :]
            y, prob = sample(logits)
            yield y, prob

    async def generate(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        moe_shard: moe_shard_pb2_grpc.MoeShardStub,
        other_drivers: list[local_driver_pb2_grpc.LocalDriverStub],
        prompt: str,
        max_tokens: int,
        temp: float,
    ) -> local_driver_pb2.UsrOutputs:
        prompt_tokens = mx.array(tokenizer.encode(prompt))

        tic = time.perf_counter()
        tokens = []
        token_strings = []
        REPLACEMENT_CHAR = "\ufffd"

        n = 0
        async for token, prob in self.generate_step(
            model,
            moe_shard,
            other_drivers,
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

        return local_driver_pb2.UsrOutputs(
            prompt_time=prompt_time,
            prompt_t_cnt=prompt_tokens.size,
            gen_time=gen_time,
            gen_t_cnt=n - 1,
            response="".join(token_strings),
        )

    async def Start(self, request: local_driver_pb2.UsrInputs, context) -> None:
        async with AsyncExitStack() as es:
            moe_shard_channel = await es.enter_async_context(
                grpc.aio.insecure_channel(
                    self.model_args.ffn_config["moe_shard_url"],
                    options=[
                        ("grpc.max_send_message_length", -1),
                        ("grpc.max_receive_message_length", -1),
                    ],
                )
            )
            moe_shard = moe_shard_pb2_grpc.MoeShardStub(moe_shard_channel)

            other_drivers = []
            for url in self.model_args.ffn_config["driver_map"]:
                if url == self.model_args.local_driver_url:
                    continue
                driver_channel = await es.enter_async_context(
                    grpc.aio.insecure_channel(
                        url,
                        options=[
                            ("grpc.max_send_message_length", -1),
                            ("grpc.max_receive_message_length", -1),
                        ],
                    )
                )
                driver = local_driver_pb2_grpc.LocalDriverStub(driver_channel)
                other_drivers.append(driver)

            response = await self.generate(
                self.model,
                self.tokenizer,
                moe_shard,
                other_drivers,
                request.prompt,
                request.max_tokens,
                DEFAULT_TEMP,
            )

        return response

    def Receive(self, request: local_driver_pb2.MoeShardOuts, context):
        layer = self.model.blocks[request.layer_num]
        layer.buffer[request.url] = {
            "expert_outs": bytes_to_mx(request.data),
            "arr_map": pickle.loads(request.arr_map),
        }

        if len(layer.buffer) == self.num_other_drivers:
            layer.sync_complete.set()

        return local_driver_pb2.Empty()


async def serve(port: int, model_path: str, config_filename: str):
    server = grpc.aio.server(
        options=[
            ("grpc.max_send_message_length", -1),
            ("grpc.max_receive_message_length", -1),
        ]
    )
    local_driver_pb2_grpc.add_LocalDriverServicer_to_server(
        LocalDriverServicer(model_path, config_filename), server
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
