#!/Users/xiangruike/miniconda3/envs/dbrx_poc/bin/python

from statistics import mean
import time

import mlx.core as mx


class MoeShard:

    def __init__(self, experts: dict) -> None:
        self.experts = experts

    def get_expert_generator(self, e: int):
        v1, w1 = None, None
        for i, weight in enumerate(self.experts[e]["weights"]):
            if i % 3 == 0:
                v1 = weight
            elif i % 3 == 1:
                w1 = weight
            else:
                yield v1, w1, weight

    def reset_expert_generators(self):
        for e in self.experts:
            self.experts[e]["generator"] = self.get_expert_generator(e)

    def warm(self) -> None:
        xs = []
        for e in self.experts:
            xs.extend(next(self.experts[e]["generator"]))

        mx.eval(mx.sum(mx.stack(xs, axis=0), axis=0))


def main():
    weights_dir = "/Users/xiangruike/dbrx-base/distributable/batch2"
    experts = {
        e: mx.load(f"{weights_dir}/expert{e}.safetensors")
        for e in [0, 1, 2, 3, 4, 5, 6, 7]
    }
    mx.eval(experts)
    shard = MoeShard(experts)

    n = 1200
    latencies = []

    for i in range(n):
        if i % 40 == 0:
            shard.reset_expert_generators()

        tic = time.perf_counter_ns()

        shard.warm()

        latency = time.perf_counter_ns() - tic
        latencies.append(latency)
        print(f"finished iter {i} in {round(latency / 1000, 3)} mu_s", flush=True)

    print(f"AVG latency: {round(mean(latencies[10:]) / 1000, 3)} mu_s")


if __name__ == "__main__":
    mx.set_default_device(mx.gpu)
    main()
