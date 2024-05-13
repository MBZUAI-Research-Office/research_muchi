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
import shard_pb2
import shard_pb2_grpc

import mlx.core as mx
import mlx.nn as nn

from numpy.random import default_rng

from serialization_utils import mx_to_bytes, bytes_to_mx

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
        self,
        shard: shard_pb2_grpc.ShardStub,
        block_num: int,
        arr_bytes: bytes,
        arr_map_bytes: bytes,
    ):
        await shard.Receive(
            shard_pb2.ShardOuts(
                url=self.url, block_num=block_num, data=arr_bytes, arr_map=arr_map_bytes
            )
        )

    async def __call__(self, inputs: mx.array, jobs: list, block_num: int) -> None:
        # sample jobs:
        # [[{14}, 1], [{}, 2]]
        # for each job,
        # job[0] indicates activated experts in this shard for inputs[i]
        # job[1] indicates num additional calculations needed to avoid
        # wire memory driver activity from surfacing

        def mlp(x, v1, w1, w2, dst):
            y = (self.act_fn(x @ w1) * (x @ v1)) @ w2
            dst.append(y)

        expert_outs = []
        arr_map = {}

        for e in self.experts:
            v1, w1, w2 = next(self.experts[e]["generator"])
            for i, x in enumerate(inputs):
                if e in jobs[i][0]:
                    mlp(x, v1, w1, w2, expert_outs)
                    arr_map[f"{i}.{e}"] = len(expert_outs) - 1
                elif jobs[i][1] > 0:
                    mlp(x, v1, w1, w2, expert_outs)
                    jobs[i][1] -= 1

        expert_outs = mx.stack(expert_outs, axis=0)
        mx.eval(expert_outs)
        arr_bytes = mx_to_bytes(expert_outs)
        arr_map_bytes = pickle.dumps(arr_map)

        tic = time.perf_counter()

        async with asyncio.TaskGroup() as tg:
            for shard in self.other_shards:
                tg.create_task(self.send(shard, block_num, arr_bytes, arr_map_bytes))

        print(f"communication took: {time.perf_counter() - tic} sec(s)", flush=True)

        return expert_outs, arr_map


class DistributedMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.expert_map = args.ffn_config["expert_map"]

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
        self,
        x: mx.array,
        shard: MoeShard,
        block_num: int,
        buffer: dict,
        sync_complete: asyncio.Event,
        inds: list[list[int]],
    ) -> mx.array:
        jobs = self.design_jobs(inds, shard.url)
        await shard(x, jobs, block_num)
        await sync_complete.wait()


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, block_num: int):
        super().__init__()
        self.ffn = DistributedMoeBlock(args)
        self.block_num = block_num
        self.buffer = {}
        self.sync_complete = asyncio.Event()

    def reset_buffer_mechanism(self):
        self.buffer = {}
        self.sync_complete.clear()

    async def __call__(
        self, x: mx.array, shard: MoeShard, inds: list[list[int]]
    ) -> mx.array:
        await self.ffn(x, shard, self.block_num, self.buffer, self.sync_complete, inds)


class DBRX(nn.Module):
    def __init__(self, args: ModelArgs, experts: mx.array):
        super().__init__()
        self.rng = default_rng(seed=0)
        self.blocks = [DecoderLayer(args, i) for i in range(args.n_layers)]
        self.moe_shard = MoeShard(args.ffn_config["shard_url"], experts)

    async def __call__(self):
        batch_size = 10
        xs = mx.random.uniform(-1, 1, (batch_size, 6144), mx.bfloat16)
        self.moe_shard.reset_expert_generators()
        for layer in self.blocks:
            inds = self.rng.choice(16, size=4, replace=False)
            await layer(xs, self.moe_shard, inds)
            layer.reset_buffer_mechanism()


class ShardServicer(shard_pb2_grpc.ShardServicer):

    def __init__(self, model_path: str, config_filename: str) -> None:
        self.model_path = Path(model_path)
        self.model_args = self.get_model_args(config_filename)
        self.model = self.load_model()
        self.num_other_shards = len(self.model_args.ffn_config["shard_map"]) - 1

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
        url = self.model_args.ffn_config["shard_url"]
        assigned_experts = self.model_args.ffn_config["shard_map"][url]
        # sample:
        # {0: {"weights": mx.array([0, 1, 2, 3])}}
        experts = {
            e: mx.load(str(self.model_path / f"expert{e}.safetensors"))
            for e in assigned_experts
        }
        mx.eval(experts)

        model = DBRX(self.model_args, experts)
        model.eval()

        return model

    async def Start(self, request: shard_pb2.Inputs, context) -> None:
        async with AsyncExitStack() as es:
            other_shards = []

            for url in self.model_args.ffn_config["other_shards"]:
                if url == self.model_args.ffn_config["shard_url"]:
                    continue
                channel = await es.enter_async_context(
                    grpc.aio.insecure_channel(
                        url,
                        options=[
                            ("grpc.max_send_message_length", -1),
                            ("grpc.max_receive_message_length", -1),
                        ],
                    )
                )
                shard = shard_pb2_grpc.ShardStub(channel)
                other_shards.append(shard)

            self.model.moe_shard.other_shards = other_shards
            await self.model()

        return shard_pb2.Outputs()

    def Receive(self, request: shard_pb2.ShardOuts, context):
        block = self.model.blocks[request.block_num]
        block.buffer[request.url] = {
            "expert_outs": bytes_to_mx(request.data),
            "arr_map": pickle.loads(request.arr_map),
        }

        if len(block.buffer) == self.num_other_shards:
            block.sync_complete.set()

        return shard_pb2.Empty()


async def serve(port: int, model_path: str, config_filename: str):
    server = grpc.aio.server(
        options=[
            ("grpc.max_send_message_length", -1),
            ("grpc.max_receive_message_length", -1),
        ]
    )
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
