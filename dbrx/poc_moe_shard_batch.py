#!/Users/xiangruike/miniconda3/envs/dbrx_poc/bin/python

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import safetensors.torch
import torch

from base import BaseModelArgs


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

        self.experts = [MLP(self.d_model, self.ffn_dim)]

    def __call__(self, x: mx.array) -> mx.array:
        return self.experts[0](x)


class DistributedDBRX(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.blocks = [
            DistributedSparseMoeBlock(args=args) for _ in range(args.n_layers)
        ]

    def __call__(self, block_num: int, inputs: mx.array) -> mx.array:
        return self.blocks[block_num](inputs)


def batch():
    expert0_path = Path(
        "~/dbrx-base/distributable/batch1/expert0.safetensors"
    ).expanduser()
    expert0_sep = safetensors.torch.load_file(expert0_path)
    arr = torch.stack(
        tuple(
            expert0_sep[f"blocks.{i}.experts.0.{j}.weight"]
            for i in range(40)
            for j in ["v1", "w1", "w2"]
        ),
        dim=0,
    )
    safetensors.torch.save_file(
        {"stacked": arr},
        Path("./test_expert_batch.safetensors"),
    )


def test0():
    arr = mx.load("/Users/xiangruike/research_muchi/dbrx/test_expert_batch.safetensors")
    mx.eval(arr)


def test1():
    arr = mx.load(
        "/Users/xiangruike/research_muchi/dbrx/test_expert_batch.safetensors"
    )["stacked"]
    weights = []
    for i in range(40):
        for j, linear_layer in enumerate(["v1", "w1", "w2"]):
            # print(arr[i * 3 + j])
            weights.append(
                (f"blocks.{i}.experts.0.{linear_layer}.weight", arr[i * 3 + j])
            )
    model_args = ModelArgs(
        d_model=6144, ffn_config={"ffn_hidden_size": 10752}, n_layers=40
    )
    model = DistributedDBRX(model_args)
    model.load_weights(weights)
    mx.eval(model.parameters())
    model.eval()


def test2():
    expert0_path = (
        "/Users/xiangruike/dbrx-base/distributable/batch1/expert0.safetensors"
    )
    model_args = ModelArgs(
        d_model=6144, ffn_config={"ffn_hidden_size": 10752}, n_layers=40
    )
    model = DistributedDBRX(model_args)
    model.load_weights(expert0_path)
    mx.eval(model.parameters())
    model.eval()


def test3():
    arr = mx.load(
        "/Users/xiangruike/research_muchi/dbrx/test_expert_batch.safetensors"
    )["stacked"]
    mx.eval(arr)
    weights = []
    for i in range(40):
        weights.append((f"blocks.{i}.experts.0.v1.weight", arr[i * 3]))
        weights.append((f"blocks.{i}.experts.0.w1.weight", arr[i * 3 + 1]))
        weights.append((f"blocks.{i}.experts.0.w2.weight", arr[i * 3 + 2].T))

    model_args = ModelArgs(
        d_model=6144, ffn_config={"ffn_hidden_size": 10752}, n_layers=40
    )
    model = DistributedDBRX(model_args)
    model.load_weights(weights)
    model.eval()

    x = mx.random.uniform(0, 0.25, (6144,), dtype=mx.bfloat16)
    for i in range(40):
        x = model(i, x)
    print(x)


def test4():
    arr = mx.load(
        "/Users/xiangruike/research_muchi/dbrx/test_expert_batch.safetensors"
    )["stacked"]
    x = mx.random.uniform(0, 0.25, (6144,), dtype=mx.bfloat16)
    for i in range(0, arr.shape[0], 3):
        x = ((x @ arr[i].T) * (x @ arr[i + 1].T)) @ arr[i + 2]
    print(x)


def test5():
    arr = mx.load(
        "/Users/xiangruike/research_muchi/dbrx/test_expert_batch.safetensors"
    )["stacked"]
    x = mx.random.uniform(0, 0.25, (6144,), dtype=mx.bfloat16)
    for i, weight in enumerate(arr):
        if i % 2 == 0:
            x = x @ weight.T
        else:
            x = x @ weight
    print(x)


def expert_enumerator():
    expert = mx.load(
        "/Users/xiangruike/research_muchi/dbrx/test_expert_batch.safetensors"
    )["stacked"]
    v1, w1 = None, None
    for i, weight in enumerate(expert):
        if i % 3 == 0:
            v1 = weight.T
        elif i % 3 == 1:
            w1 = weight.T
        else:
            yield v1, w1, weight

def test6():
    x = mx.random.uniform(0, 0.25, (6144,), dtype=mx.bfloat16)
    for v1, w1, w2 in expert_enumerator():
        x = ((x @ v1) * (x @ w1)) @ w2
    print(x)


if __name__ == "__main__":
    # mx.metal.set_cache_limit(0)
    # batch()
    # test0()
    # test1()
    # test2()
    # test3()
    # test4()
    # test5()
    test6()
