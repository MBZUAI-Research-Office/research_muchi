#!/Users/xiangruike/miniconda3/envs/dbrx_poc/bin/python

from dataclasses import dataclass
from pathlib import Path
import inspect
import json
import logging
import mlx.core as mx
import mlx.nn as nn


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


class RawWeights:

    def __init__(
        self,
        n_layers: int,
        experts: dict,
    ) -> None:
        ptrs = {i: {} for i in range(n_layers)}
        for e, d in experts.items():
            for j, mat in enumerate(d["weights"]):
                i = j // 3
                if e not in ptrs[i]:
                    ptrs[i][e] = {}
                if j % 3 == 0:
                    ptrs[i][e]["v1"] = mat
                elif j % 3 == 1:
                    ptrs[i][e]["w1"] = mat
                else:
                    ptrs[i][e]["w2"] = mat

        e_warmup = []
        for e in experts:
            for vec in ptrs[0][e]["v1"]:
                e_warmup.append(vec)
                break

        self.ptrs = ptrs
        self.e_warmup = e_warmup

    def __call__(self, k):
        return self.ptrs[k]


class Test:

    def __init__(self, model_path: str) -> None:
        self.act_fn = nn.silu
        self.n_layers = 40
        self.model_path = Path(model_path)
        self.raw_weights = self.load_model()

    def load_model(self):
        experts = {
            e: mx.load(str(self.model_path / f"expert{e}.safetensors"))
            for e in [0, 1, 2, 3]
        }
        mx.eval(experts)
        return RawWeights(self.n_layers, experts)

    def mlp(self, x, job, li):
        ws = self.raw_weights(li)
        expert_outs, cs = [], []
        for e in job:
            y = (self.act_fn(x @ ws[e]["w1"].T) * (x @ ws[e]["v1"].T)) @ ws[e]["w2"]
            expert_outs.append(y)
            cs.append(job[e])

        y = (mx.stack(expert_outs, axis=-1) * mx.stack(cs, axis=0)).sum(axis=-1)
        mx.eval(y)

    # def mlp(self, x, job, li):
    #     ws = self.raw_weights(li)
    #     expert_outs = []
    #     for e in job:
    #         y = (self.act_fn(x @ ws[e]["w1"].T) * (x @ ws[e]["v1"].T)) @ ws[e]["w2"]
    #         expert_outs.append(y)

    #     mx.eval(mx.stack(expert_outs, axis=0))

    def test(self):
        x = mx.ones((6144,), dtype=mx.bfloat16)
        mx.eval(x)
        job = {0: mx.array(1, dtype=mx.bfloat16)}

        for _ in range(self.n_layers):
            mx.eval(mx.sum(mx.stack(self.raw_weights.e_warmup, axis=0), axis=0))

        for li in range(self.n_layers):
            self.mlp(x, job, li)

        print("warmup finished, started testing...", flush=True)

        for li in range(self.n_layers):
            self.mlp(x, job, li)


if __name__ == "__main__":
    test = Test("/Users/xiangruike/dbrx-instruct/distributable/batch2")
    test.test()
