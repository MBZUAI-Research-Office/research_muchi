#!/Users/xiangruike/miniconda3/envs/dbrx_poc/bin/python

from statistics import mean
import time
import gc

import numpy as np
import mlx.core as mx


def test0():
    weights_dir = "/Users/xiangruike/dbrx-instruct/distributable/batch2"
    non_expert = mx.load(f"{weights_dir}/non-expert.safetensors")
    target = {"Wqkv": [], "out_proj": []}

    for k, w in non_expert.items():
        for tar in target:
            if tar in k:
                target[tar].append(w)
    
    for tar, ws in target.items():
        mx.eval(ws)
        print(f"{tar} has #{len(ws)} weights")

    n = 1000
    warmup = 10
    t_wait = 0.002

    for tar, ws in target.items():
        for _ in range(warmup):
            for mat in ws:
                for vec in mat:
                    mx.eval(vec + 1)
                    break

        tic = time.perf_counter_ns()

        for _ in range(n):
            for mat in ws:
                for vec in mat:
                    mx.eval(vec + 1)
                    break
                time.sleep(t_wait)

        latency = time.perf_counter_ns() - tic
        latency -= t_wait * len(ws) * n * 1000**3
        print(f"{tar} avg latency: {round(latency / len(ws) / n / 1000**2, 3)} ms")


def test1():
    np_arr = np.ones(shape=(40, 8192, 6144), dtype=np.float32)
    stacked_qkv = mx.array(np_arr, dtype=mx.bfloat16)
    x = mx.ones((1, 8192), dtype=mx.bfloat16)
    mx.eval(stacked_qkv, x)
    gc.collect()

    n = 10000
    t_wait = 0  # sec

    # warmup
    for _ in range(10):
        for mat in stacked_qkv:
            for vec in mat:
                mx.eval(vec + 1)
                break
            break

    tic = time.perf_counter_ns()

    for _ in range(n):
        for mat in stacked_qkv:
            for vec in mat:
                mx.eval(vec + 1, mx.stack([x @ mat for _ in range(20)], axis=0))
                break
            break
        time.sleep(t_wait)

    latency = time.perf_counter_ns() - tic
    latency -= t_wait * 1000**3 * n
    print(f"latency: {round(latency / n / 1000**2, 3)} ms")


if __name__ == "__main__":
    mx.set_default_device(mx.gpu)
    test1()
