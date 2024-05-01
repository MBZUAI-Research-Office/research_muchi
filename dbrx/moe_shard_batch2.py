#!/Users/xiangruike/miniconda3/envs/dbrx_poc/bin/python

from concurrent import futures
from pathlib import Path
import argparse
import json
import logging

import grpc
import moe_shard_pb2
import moe_shard_pb2_grpc

import numpy as np

import mlx.core as mx
import mlx.nn as nn


class DistributedDBRX:

    def __init__(self, experts: dict) -> None:
        self.experts = experts
        self.expert_generators = self.get_expert_generators()
        self.act_fn = nn.silu

    def get_expert_generator(self, expert: mx.array):
        v1, w1 = None, None
        for i, weight in enumerate(expert):
            if i % 3 == 0:
                v1 = weight.T
            elif i % 3 == 1:
                w1 = weight.T
            else:
                yield v1, w1, weight

    def get_expert_generators(self):
        return [self.get_expert_generator(e) for e in self.experts]

    def next_safe(self, e):
        try:
            return next(self.expert_generators[e])
        except StopIteration:
            self.expert_generators = self.get_expert_generators()
            return next(self.expert_generators[e])

    def __call__(self, inputs: np.array) -> np.array:
        x = mx.array(inputs, dtype=mx.bfloat16)
        ys = []
        for e in range(len(self.expert_generators)):
            v1, w1, w2 = self.next_safe(e)
            ys.append((self.act_fn(x @ w1) * (x @ v1)) @ w2)

        # conversion is needed since NumPy does not support bfloat16 arrays
        # see: https://ml-explore.github.io/mlx/build/html/usage/numpy.html
        return np.array(mx.array(ys).astype(mx.float32))


class MoeShardServicer(moe_shard_pb2_grpc.MoeShardServicer):

    def __init__(self, model_path: str, config_filename: str) -> None:
        self.model_path = Path(model_path)
        self.model = self.load_model(config_filename)

    def Execute(self, request: moe_shard_pb2.Inputs, context):
        outputs = self.model(np.frombuffer(request.data, dtype=np.float32))
        return moe_shard_pb2.Outputs(data=outputs.tobytes())

    def load_model(self, config_filename: str):
        try:
            with open(self.model_path / config_filename, "r") as f:
                config = json.load(f)
        except FileNotFoundError:
            logging.error(f"{config_filename} not found in {self.model_path}")
            raise

        experts = [
            mx.load(str(self.model_path / f"expert{e}.safetensors"))["weights"]
            for e in config["ffn_config"]["assigned_experts"]
        ]
        mx.eval(experts)

        return DistributedDBRX(experts)


def serve(port: int, model_path: str, config_filename: str):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    moe_shard_pb2_grpc.add_MoeShardServicer_to_server(
        MoeShardServicer(model_path, config_filename), server
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
