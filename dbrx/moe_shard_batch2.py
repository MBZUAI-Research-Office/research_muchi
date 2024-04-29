#!/Users/xiangruike/miniconda3/envs/dbrx_poc/bin/python

from pathlib import Path
import argparse
import asyncio
import json
import logging
import pickle

import grpc
import moe_shard_pb2
import moe_shard_pb2_grpc

import numpy as np

import mlx.core as mx
import mlx.nn as nn

# coroutines to be invoked when the event loop is shutting down
# copied from:
# https://github.com/grpc/grpc/blob/master/examples/python/helloworld/async_greeter_server_with_graceful_shutdown.py
_cleanup_coroutines = []


class DistributedDBRX:

    def __init__(self, experts: dict) -> None:
        self.experts = experts
        self.expert_generators = None
        self.act_fn = nn.silu
        self.reset_generators()

    def get_expert_generator(self, expert: mx.array):
        v1, w1 = None, None
        for i, weight in enumerate(expert):
            if i % 3 == 0:
                v1 = weight.T
            elif i % 3 == 1:
                w1 = weight.T
            else:
                yield v1, w1, weight

    def reset_generators(self):
        self.expert_generators = {
            k: self.get_expert_generator(v) for k, v in self.experts.items()
        }

    def next_safe(self, e):
        try:
            return next(self.expert_generators[e])
        except StopIteration:
            self.reset_generators()
            return next(self.expert_generators[e])

    def to_next_block(self) -> None:
        # in case that this shard is not activated for this block/layer
        for e in self.experts:
            self.next_safe(e)

    def __call__(self, activated_experts: set, inputs: np.array) -> np.array:
        x = mx.array(inputs, dtype=mx.bfloat16)
        ys = []
        for e in self.experts:
            if e in activated_experts:
                v1, w1, w2 = self.next_safe(e)
                ys.append((self.act_fn(x @ w1) * (x @ v1)) @ w2)
            else:
                # ensures that expert_generators are in sync
                self.next_safe(e)

        # conversion is needed since NumPy does not support bfloat16 arrays
        # see: https://ml-explore.github.io/mlx/build/html/usage/numpy.html
        return np.array(mx.array(ys).astype(mx.float32))


class MoeShardServicer(moe_shard_pb2_grpc.MoeShardServicer):

    def __init__(self, model_path: str, config_filename: str) -> None:
        self.model_path = Path(model_path)
        self.model = self.load_model(config_filename)

    async def Execute(
        self, request: moe_shard_pb2.Inputs, context: grpc.aio.ServicerContext
    ):
        # DEV
        print(f"start executing", flush=True)
        print("-" * 10, flush=True)

        outputs = self.model(
            pickle.loads(request.activated_experts),
            np.frombuffer(request.data, dtype=np.float32),
        )

        # DEV
        print("done executing", flush=True)
        print("-" * 10, flush=True)

        return moe_shard_pb2.Outputs(data=outputs.tobytes())

    async def ToNextBlock(
        self, request: moe_shard_pb2.Inputs, context: grpc.aio.ServicerContext
    ):
        self.model.to_next_block()
        return moe_shard_pb2.Empty()

    def load_model(self, config_filename: str):
        try:
            with open(self.model_path / config_filename, "r") as f:
                config = json.load(f)
        except FileNotFoundError:
            logging.error(f"{config_filename} not found in {self.model_path}")
            raise

        experts = {
            e: mx.load(str(self.model_path / f"expert{e}.safetensors"))["weights"]
            for e in config["ffn_config"]["assigned_experts"]
        }
        mx.eval(experts)

        return DistributedDBRX(experts)


async def serve(port: int, model_path: str, config_filename: str):
    server = grpc.aio.server()
    moe_shard_pb2_grpc.add_MoeShardServicer_to_server(
        MoeShardServicer(model_path, config_filename), server
    )
    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)
    logging.info(f"Starting server on {listen_addr}")
    await server.start()

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

    # mx.metal.set_cache_limit(0)
    logging.basicConfig(level=logging.INFO)
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(serve(args.port, args.model_path, args.config_filename))
    finally:
        loop.run_until_complete(*_cleanup_coroutines)
        loop.close()
