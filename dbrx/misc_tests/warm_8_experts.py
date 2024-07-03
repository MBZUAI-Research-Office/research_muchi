#!/Users/xiangruike/miniconda3/envs/dbrx_poc/bin/python

from statistics import mean
import time

import mlx.core as mx


class MoeShard:

    def __init__(self, experts: dict) -> None:
        self.experts = experts

    def get_ptrs(self) -> None:
        for e in self.experts:
            for mat in self.experts[e]["weights"]:
                for vec in mat:
                    self.experts[e]["ptr"] = vec
                    break
                break

    def warm(self) -> None:
        xs = [e["ptr"] for e in self.experts.values()]
        mx.eval(mx.sum(mx.stack(xs, axis=0), axis=0))


def main():
    weights_dir = "/Users/xiangruike/dbrx-instruct/distributable/batch2"
    experts = {
        e: mx.load(f"{weights_dir}/expert{e}.safetensors")
        for e in [0, 1, 2, 3, 4, 5, 6, 7]
    }
    mx.eval(experts)
    shard = MoeShard(experts)

    n = 100000
    warmup = 10
    shard.get_ptrs()

    for _ in range(warmup):
        shard.warm()

    tic = time.perf_counter_ns()

    for _ in range(n):
        shard.warm()

    toc = time.perf_counter_ns()
    print(f"AVG latency: {round((toc - tic) / n / 1000**2, 3)} ms")


if __name__ == "__main__":
    mx.set_default_device(mx.gpu)
    main()
