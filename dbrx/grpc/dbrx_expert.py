#!/Users/xiangruike/miniconda3/envs/dbrx_poc/bin/python

from dataclasses import dataclass
from pathlib import Path
import argparse
import asyncio
import json
import logging

import grpc
import test_expert_pb2
import test_expert_pb2_grpc

import numpy as np

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    d_model: int
    ffn_config: dict
    n_layers: int


class MLP(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int):
        super().__init__()
        self.v1 = nn.Linear(d_model, ffn_dim, bias=False)
        self.w1 = nn.Linear(d_model, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, d_model, bias=False)
        self.act_fn = nn.silu

    def __call__(self, x: mx.array) -> mx.array:
        current_hidden_states = self.act_fn(self.w1(x)) * self.v1(x)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states


class DistributedSparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.d_model = args.d_model
        self.ffn_dim = args.ffn_config["ffn_hidden_size"]
        self.assigned_experts = args.ffn_config["assigned_experts"]

        self.experts = {
            i: MLP(self.d_model, self.ffn_dim) for i in self.assigned_experts
        }

    def __call__(self, activated_experts: np.array, x: mx.array) -> mx.array:
        return mx.array([self.experts[i](x) for i in activated_experts])


class DistributedDBRX(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.blocks = [
            DistributedSparseMoeBlock(args=args) for _ in range(args.n_layers)
        ]

    def __call__(
        self, block_num: int, activated_experts: np.array, inputs: np.array
    ) -> np.array:
        # conversion is needed since NumPy does not support bfloat16 arrays
        # see: https://ml-explore.github.io/mlx/build/html/usage/numpy.html
        return np.array(
            self.blocks[block_num](
                activated_experts, mx.array(inputs, dtype=mx.bfloat16)
            ).astype(mx.float32)
        )


class ExpertServicer(test_expert_pb2_grpc.ExpertServicer):

    def __init__(self, model_path: str) -> None:
        self.model_path = Path(model_path)
        self.model = self.load_model()

    async def Execute(
        self, request: test_expert_pb2.Input, context: grpc.aio.ServicerContext
    ):
        outputs = self.model(
            request.block_num,
            np.frombuffer(request.activated_experts, dtype=np.int64),
            np.frombuffer(request.data, dtype=np.float32),
        )
        return test_expert_pb2.Output(data=outputs.tobytes())

    def load_model(self) -> nn.Module:
        try:
            with open(self.model_path / "config.json", "r") as f:
                config = json.load(f)
        except FileNotFoundError:
            logging.error(f"Config file not found in {self.model_path}")
            raise

        model_args = ModelArgs.from_dict(config)
        model = DistributedDBRX(model_args)

        weights = {}
        for i in model_args.ffn_config["assigned_experts"]:
            weights.update(mx.load(str(self.model_path / f"expert{i}.npz")))

        model.load_weights(list(weights.items()))
        mx.eval(model.parameters())
        model.eval()

        return model


async def serve(port: int, model_path: str):
    server = grpc.aio.server()
    test_expert_pb2_grpc.add_ExpertServicer_to_server(ExpertServicer(model_path), server)
    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)
    logging.info("Starting server on %s", listen_addr)
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument("--model-path", type=str)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve(args.port, args.model_path))
