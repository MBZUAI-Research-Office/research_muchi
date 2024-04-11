from pathlib import Path
import argparse
import asyncio
import glob
import logging
import json

import numpy as np

import grpc
import test_expert_pb2
import test_expert_pb2_grpc

import mlx.core as mx
import mlx.nn as nn


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

class Weights:

    def __init__(self, model_path: str) -> None:
        self.model_path = Path(model_path)
    
def load_model(model_path: Path) -> nn.Module:
    try:
        with open(model_path / "config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        logging.error(f"Config file not found in {model_path}")
        raise

    weight_files = glob.glob(str(model_path / "*.safetensors"))
    if not weight_files:
        logging.error(f"No safetensors found in {model_path}")
        raise FileNotFoundError(f"No safetensors found in {model_path}")

    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    model_class, model_args_class = _get_classes(config=config)

    model_args = model_args_class.from_dict(config)
    model = model_class(model_args)

    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)


    model.load_weights(list(weights.items()))

    if not lazy:
        mx.eval(model.parameters())

    model.eval()
    return model

class ExpertServicer(test_expert_pb2_grpc.ExpertServicer):

    async def Execute(
        self, request: test_expert_pb2.Input, context: grpc.aio.ServicerContext
    ):
        arr = np.frombuffer(request.data, dtype=np.float16)
        return test_expert_pb2.Output(data=arr.tobytes())


async def serve(port: int):
    server = grpc.aio.server()
    test_expert_pb2_grpc.add_ExpertServicer_to_server(ExpertServicer(), server)
    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)
    logging.info("Starting server on %s", listen_addr)
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=3000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve(args.port))
