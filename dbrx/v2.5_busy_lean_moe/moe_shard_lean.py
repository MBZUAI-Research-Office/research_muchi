#!/Users/xiangruike/miniconda3/envs/dbrx_poc/bin/python

from concurrent import futures
from pathlib import Path
import argparse
import json
import logging
import pickle

import grpc
import moe_shard_lean_pb2
import moe_shard_lean_pb2_grpc

import mlx.core as mx
import mlx.nn as nn


from serialization_utils import mx_to_bytes, bytes_to_mx


class MoeShard:

    def __init__(
        self,
        experts: dict,
    ) -> None:
        self.experts = experts
        self.act_fn = nn.silu
        self.reset_expert_generators()

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

    def next_safe(self, e):
        try:
            return next(self.experts[e]["generator"])
        except StopIteration:
            self.reset_expert_generators()
            return next(self.experts[e]["generator"])

    def __call__(self, xs: mx.array, jobs: list) -> tuple[mx.array, dict]:
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
            v1, w1, w2 = self.next_safe(e)
            for i, x in enumerate(xs):
                if e in jobs[i][0]:
                    mlp(x, v1, w1, w2, expert_outs)
                    arr_map[f"{i}.{e}"] = len(expert_outs) - 1
                elif jobs[i][1] > 0:
                    mlp(x, v1, w1, w2, expert_outs)
                    jobs[i][1] -= 1

        expert_outs = mx.stack(expert_outs, axis=0)
        mx.eval(expert_outs)
        return expert_outs, arr_map


class MoeShardServicer(moe_shard_lean_pb2_grpc.MoeShardServicer):

    def __init__(self, model_path: str, config_filename: str) -> None:
        self.model_path = Path(model_path)
        self.model_args = self.get_model_args(config_filename)
        self.model = self.load_model()

    def Execute(self, request: moe_shard_lean_pb2.Inputs, context):
        ys, arr_map = self.model(bytes_to_mx(request.data), pickle.loads(request.jobs))
        return moe_shard_lean_pb2.Outputs(
            data=mx_to_bytes(ys), arr_map=pickle.dumps(arr_map)
        )

    def get_model_args(self, config_filename: str) -> dict:
        try:
            with open(self.model_path / config_filename, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logging.error(f"{config_filename} not found in {self.model_path}")
            raise

    def load_model(self):
        assigned_experts = self.model_args["ffn_config"]["assigned_experts"]
        # sample:
        # {0: {"weights": mx.array([0, 1, 2, 3])}}
        experts = {
            e: mx.load(str(self.model_path / f"expert{e}.safetensors"))
            for e in assigned_experts
        }
        mx.eval(experts)

        return MoeShard(experts)


def serve(port: int, model_path: str, config_filename: str):
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_send_message_length", -1),
            ("grpc.max_receive_message_length", -1),
        ],
    )
    moe_shard_lean_pb2_grpc.add_MoeShardServicer_to_server(
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
