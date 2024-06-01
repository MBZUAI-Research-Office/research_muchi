#!/Users/xiangruike/miniconda3/envs/dbrx_poc/bin/python

from statistics import mean
import time

import mlx.core as mx
import mlx.nn as nn

from numpy.random import default_rng

EXPERT_MAP = {
    0: "192.168.1.2:2000",
    1: "192.168.1.2:2000",
    2: "192.168.1.2:2000",
    3: "192.168.1.2:2000",
    4: "192.168.1.4:4000",
    5: "192.168.1.4:4000",
    6: "192.168.1.4:4000",
    7: "192.168.1.4:4000",
    8: "192.168.1.5:5000",
    9: "192.168.1.5:5000",
    10: "192.168.1.5:5000",
    11: "192.168.1.5:5000",
    12: "192.168.1.6:6000",
    13: "192.168.1.6:6000",
    14: "192.168.1.6:6000",
    15: "192.168.1.6:6000",
}
MY_URL = "192.168.1.6:6000"


class MoeShard:

    def __init__(self, experts: dict) -> None:
        self.experts = experts
        self.act_fn = nn.silu
        self.ptr_cache = {}

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

    def __call__(
        self, x: mx.array, job: tuple, use_cache: bool
    ) -> tuple[mx.array, dict]:
        # sample job:
        # ({14}, 1)
        # job[0] indicates activated experts in this shard for x
        # job[1] indicates num additional calculations needed to avoid
        # wire memory driver activity from surfacing

        def get_weights(e):
            if not use_cache:
                self.ptr_cache[e] = next(self.experts[e]["generator"])

            return self.ptr_cache[e]

        def mlp(x, v1, w1, w2, dst):
            y = (self.act_fn(x @ w1) * (x @ v1)) @ w2
            dst.append(y)

        expert_outs = []
        arr_map = {}

        for e in self.experts:
            v1, w1, w2 = get_weights(e)
            if e in job[0]:
                mlp(x, v1, w1, w2, expert_outs)
                arr_map[e] = len(expert_outs) - 1
            elif job[1] > 0:
                mlp(x, v1, w1, w2, expert_outs)
                job[1] -= 1
        
        if len(expert_outs) == 0:
            return

        expert_outs = mx.stack(expert_outs, axis=0)
        mx.eval(expert_outs)

        return expert_outs, arr_map


def design_jobs(inds: list[list[int]]) -> list:
    jobs = []

    for activated_experts in inds:
        job = set()
        shard_loads = {}

        for e in activated_experts:
            url = EXPERT_MAP[e]
            if url == MY_URL:
                job.add(e)
            shard_loads[url] = shard_loads.get(url, 0) + 1

        # jobs.append((job, max(shard_loads.values()) - len(job)))
        jobs.append((job, 0))

    return jobs


def main():
    weights_dir = "/Users/xiangruike/dbrx-base/distributable/batch2"
    experts = {
        e: mx.load(f"{weights_dir}/expert{e}.safetensors") for e in [12, 13, 14, 15]
    }
    mx.eval(experts)
    shard = MoeShard(experts)

    batch_size = 64
    n_layers = 400

    rng = default_rng(seed=0)
    jobss = []
    for _i in range(n_layers):
        inds = []
        for _j in range(batch_size):
            inds.append(rng.choice(16, size=4, replace=False).tolist())
        jobss.append(design_jobs(inds))

    xs = mx.random.uniform(-1, 1, (6144,), mx.bfloat16)
    mx.eval(xs)

    latencies = []

    for i in range(n_layers):
        if i % 40 == 0:
            shard.reset_expert_generators()

        tic = time.perf_counter_ns()

        for j in range(batch_size):
            shard(xs, jobss[i][j], use_cache=bool(j > 0))

        latency = time.perf_counter_ns() - tic
        latencies.append(latency)
        print(f"finished layer {i} in {round(latency / 1000, 3)} mu_s", flush=True)
        time.sleep(0.002)

    print(f"AVG latency: {round(mean(latencies[5:]) / 1000, 3)} mu_s")


if __name__ == "__main__":
    main()
