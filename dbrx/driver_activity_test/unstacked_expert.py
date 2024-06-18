#!/Users/xiangruike/miniconda3/envs/dbrx_poc/bin/python

from dataclasses import dataclass
from pathlib import Path
import argparse
import inspect
import json
import logging
import time

from numpy.random import default_rng

import mlx.core as mx
import mlx.nn as nn


@dataclass
class ModelArgs:
    d_model: int
    ffn_config: dict
    n_layers: int

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


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
        # because nn.Module.load_weights does not work with dicts : (
        self.expert_to_i = args.ffn_config["expert_to_i"]

        self.experts = [MLP(self.d_model, self.ffn_dim) for _ in self.expert_to_i]

    def __call__(self, activated_experts: mx.array, x: mx.array) -> mx.array:
        return mx.stack(
            [self.experts[self.expert_to_i[e]](x) for e in activated_experts], axis=0
        )


class DistributedDBRX(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.blocks = [
            DistributedSparseMoeBlock(args=args) for _ in range(args.n_layers)
        ]

    def __call__(
        self, block_num: int, activated_experts: list, inputs: mx.array
    ) -> mx.array:
        if len(activated_experts) > 0:
            return self.blocks[block_num](activated_experts, inputs)
        return mx.array(False)


class SimpleMLP:

    def __init__(self, ei: int, weights: dict) -> None:
        self.ei = ei
        self.weights = weights

    def __call__(self, li: int, x: mx.array) -> mx.array:

        def W(wi: str) -> mx.array:
            return self.weights[f"blocks.{li}.experts.{self.ei}.{wi}.weight"]

        return ((x @ W("w1").T) * (x @ W("v1").T)) @ W("w2")


class Test:

    def __init__(self, model_path: str, config_filename: str) -> None:
        self.model_path = Path(model_path)
        self.model = self.load_model(config_filename)

    def load_model(self, config_filename: str) -> DistributedDBRX:
        try:
            with open(self.model_path / config_filename, "r") as f:
                config = json.load(f)
        except FileNotFoundError:
            logging.error(f"{config_filename} not found in {self.model_path}")
            raise

        model_args = ModelArgs.from_dict(config)

        model_args.ffn_config["expert_to_i"] = {}
        weights = {}
        for i, e in enumerate(model_args.ffn_config["assigned_experts"]):
            model_args.ffn_config["expert_to_i"][e] = i

            for k, v in mx.load(
                str(self.model_path / f"expert{e}.safetensors")
            ).items():
                # sample k: blocks.10.experts.12.v1.weight
                # change expert number (0 - 15) to internal index
                k_splits = k.split(".")
                k_splits[3] = str(i)
                weights[".".join(k_splits)] = v.T if k_splits[4] == "w2" else v

        del model_args.ffn_config["assigned_experts"]

        model = DistributedDBRX(model_args)
        model.load_weights(list(weights.items()))
        mx.eval(model.parameters())
        model.eval()

        return model

    def start_0(self):
        n_tokens = 10
        n_layers = 40
        x = mx.ones((1, 6144), dtype=mx.bfloat16)
        mx.eval(x)

        for i in range(n_tokens):
            tic = time.perf_counter_ns()

            for j in range(n_layers):
                mx.eval(self.model(j, [8], x))

            toc = time.perf_counter_ns()
            print(f"avg latency per token: {(toc - tic) / 1000**2} ms")
            time.sleep(1)

    def start_1(self):
        n_tokens = 256
        n_layers = 40
        x = mx.ones((1, 6144), dtype=mx.bfloat16)
        mx.eval(x)
        assigned_experts = {8, 9, 10, 11, 12, 13, 14, 15}
        selection_stats = {e: 0 for e in assigned_experts}

        rng = default_rng(seed=0)
        jobs = []
        for _i in range(n_tokens):
            by_layer = []
            for _j in range(n_layers):
                router_selection = rng.choice(16, size=4, replace=False)
                job = []
                for e in router_selection:
                    if e in assigned_experts:
                        job.append(e)
                        selection_stats[e] += 1
                by_layer.append(job)
            jobs.append(by_layer)

        tic = time.perf_counter_ns()

        for i in range(n_tokens):
            for j in range(n_layers):
                mx.eval(self.model(j, jobs[i][j], x))

        toc = time.perf_counter_ns()
        print(f"avg latency per token: {(toc - tic) / (n_tokens * 1000**2)} ms")
        print("selection stats:")
        print(selection_stats)


class SimpleTest:

    def __init__(self, model_path: str, ei: int) -> None:
        self.model_path = Path(model_path)
        self.model = self.load_model(ei)

    def load_model(self, ei: int) -> SimpleMLP:
        weights = mx.load(str(self.model_path / f"expert{ei}.safetensors"))
        mx.eval(weights)
        return SimpleMLP(ei, weights)

    def start(self) -> None:
        n_tokens = 10
        n_layers = 40
        x = mx.ones((1, 6144), dtype=mx.bfloat16)
        mx.eval(x)

        for i in range(n_tokens):
            tic = time.perf_counter_ns()

            for j in range(n_layers):
                mx.eval(self.model(j, x))

            toc = time.perf_counter_ns()
            print(f"avg latency per token: {(toc - tic) / 1000**2} ms")
            time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--config-filename", type=str)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    # test = Test(args.model_path, args.config_filename)
    # test.start_0()

    test = SimpleTest(args.model_path, 0)
    test.start()
