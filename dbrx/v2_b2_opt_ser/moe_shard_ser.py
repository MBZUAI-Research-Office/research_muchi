#!/Users/xiangruike/miniconda3/envs/dbrx_poc/bin/python

from concurrent import futures
from pathlib import Path
import argparse
import json
import logging
import time

import grpc
import moe_shard_ser_pb2
import moe_shard_ser_pb2_grpc

import mlx.core as mx
import mlx.nn as nn

from serialization_utils import mx_to_bytes, bytes_to_mx


class DistributedDBRX:

    def __init__(self, experts: list) -> None:
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

    def __call__(self, inputs: mx.array) -> mx.array:
        expert_outs = []

        for e in range(len(self.expert_generators)):
            v1, w1, w2 = self.next_safe(e)
            ys = []
            for x in inputs:
                ys.append((self.act_fn(x @ w1) * (x @ v1)) @ w2)
            expert_outs.append(mx.stack(ys, axis=0))

        res = mx.stack(expert_outs, axis=0)
        return mx.swapaxes(res, 0, 1)


class MoeShardServicer(moe_shard_ser_pb2_grpc.MoeShardServicer):

    def __init__(self, model_path: str, config_filename: str) -> None:
        self.model_path = Path(model_path)
        self.model_args = self.get_model_args(config_filename)
        self.model = self.load_model()

    # def Execute(self, request: moe_shard_ser_pb2.Inputs, context):
    #     inputs = bytes_to_mx(
    #         request.data, (request.batch_size, self.model_args["d_model"])
    #     )
    #     outputs = self.model(inputs)
    #     return moe_shard_ser_pb2.Outputs(data=mx_to_bytes(outputs))

    def Execute(self, request: moe_shard_ser_pb2.Inputs, context):
        tic = time.perf_counter_ns()
        inputs = bytes_to_mx(request.data)
        outputs = self.model(inputs)
        return moe_shard_ser_pb2.Outputs(
            data=mx_to_bytes(outputs), start=tic, end=time.perf_counter_ns()
        )

    def get_model_args(self, config_filename: str) -> dict:
        try:
            with open(self.model_path / config_filename, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logging.error(f"{config_filename} not found in {self.model_path}")
            raise

    def load_model(self):
        experts = [
            mx.load(str(self.model_path / f"expert{e}.safetensors"))["weights"]
            for e in self.model_args["ffn_config"]["assigned_experts"]
        ]
        mx.eval(experts)

        return DistributedDBRX(experts)


def serve(port: int, model_path: str, config_filename: str):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    moe_shard_ser_pb2_grpc.add_MoeShardServicer_to_server(
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
