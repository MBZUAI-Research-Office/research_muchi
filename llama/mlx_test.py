#!/Users/xiangruike/miniconda3/envs/mlx_custom/bin/python

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import mlx.core as mx


class RepresentativeWorkload:

    DEFAULT_INPUT_M = 32
    DEFAULT_SCALING_FACTOR = 10**-5

    def __init__(self, args) -> None:
        self.args = args
        # mlx cannot work with Path objects : (
        self.weights_path = self.args.weights_path
        self.output_path = self.args.output_path
        self.log_path = Path(self.args.log_path)
        self.latencies = {}

        # determined either when weights are initialized or loaded
        self.required_init = True
        self.mat_dim = None
        self.load_sep = None
        self.num_weights = None

        self.init_weights()
        self.weights = self.load_weights()
        self.print_weights_info()

    def print_barrier(self):
        print("=" * 20)

    def print_weights_info(self):
        print(f"using weights from {self.weights_path}")
        print(f"mat dim: {self.mat_dim}")
        print(f"num weights: {self.num_weights}")
        print("last layer:")
        if self.load_sep:
            print(self.weights[f"arr{len(self.weights) - 1}"])
        else:
            print(self.weights[-1])
        self.print_barrier()

    def init_weights(self):
        if Path(self.weights_path).is_file():
            self.required_init = False
            return

        self.mat_dim = self.args.mat_dim
        self.load_sep = self.args.load_sep

        weights_size = (4 * self.mat_dim**2) / 1024**3
        assert self.args.data_size % weights_size == 0

        self.num_weights = int(self.args.data_size / weights_size)

        tic = time.perf_counter()

        if self.load_sep:
            weights = {}
            for i in range(self.num_weights):
                weights[f"arr{i}"] = (
                    np.ones(shape=(self.mat_dim, self.mat_dim), dtype=np.float32)
                    * self.DEFAULT_SCALING_FACTOR
                    * (i + 1)
                )
            np.savez(self.weights_path, **weights)
        else:
            weights = (
                np.ones(
                    shape=(self.num_weights, self.mat_dim, self.mat_dim),
                    dtype=np.float32,
                )
                * self.DEFAULT_SCALING_FACTOR
            )
            for i in range(self.num_weights):
                weights[i] *= i + 1
            np.save(self.weights_path, weights)

        print(f"WEIGHTS INIT LATENCY: {time.perf_counter() - tic} seconds", flush=True)
        self.print_barrier()

        # forces python to free up memory
        del weights
        gc.collect()

    def load_weights(self):
        tic = time.perf_counter()

        weights = mx.load(self.weights_path)
        mx.eval(weights)

        self.latencies["load_latency"] = time.perf_counter() - tic
        print(f"LOAD LATENCY: {self.latencies['load_latency']} seconds", flush=True)
        self.print_barrier()

        if not self.required_init:
            # mlx loads arrays saved with np.savez as a dictionary of names to arrays
            self.load_sep = isinstance(weights, dict)

            if self.load_sep:
                self.mat_dim = weights["arr0"].shape[0]
                self.num_weights = len(weights)
            else:
                self.mat_dim = weights.shape[1]
                self.num_weights = weights.shape[0]

        return weights

    def execute_step(self, save_output: bool) -> float:
        token = (
            mx.ones(shape=(self.DEFAULT_INPUT_M, self.mat_dim))
            * self.DEFAULT_SCALING_FACTOR
        )

        tic = time.perf_counter()

        if self.load_sep:
            for i in range(self.num_weights):
                token = mx.matmul(token, self.weights[f"arr{i}"])
                # clips output to prevent growth to inf
                if token[0, 0] > 1:
                    token = 1 / token
        else:
            for layer in self.weights:
                token = mx.matmul(token, layer)
                # clips output to prevent growth to inf
                if token[0, 0] > 1:
                    token = 1 / token
        mx.eval(token)

        toc = time.perf_counter()

        if save_output:
            mx.save(self.output_path, token)
            print("last sample's output:")
            print(token)
            self.print_barrier()

        return toc - tic

    def execute(self, n_samples: int) -> None:
        total_latency = 0
        for i in range(n_samples):
            save_output = True if i == n_samples - 1 else False
            latency = self.execute_step(save_output)
            total_latency += latency

        self.latencies["avg_compute_latency"] = total_latency / n_samples
        self.latencies["total_compute_latency"] = total_latency
        print(f"AVERAGE LATENCY: {self.latencies['avg_compute_latency']} seconds")
        print(f"TOTAL LATENCY: {total_latency} seconds")
        self.print_barrier()

        self.log_to_file()

    def log_to_file(self) -> None:
        with open(self.log_path, "a") as logs:
            logs.write(
                f"{self.latencies['load_latency']},{self.latencies['avg_compute_latency']},{self.latencies['total_compute_latency']}\n"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="a simple yet memory demanding workload"
    )
    parser.add_argument(
        "--data-size",
        type=int,
        default=1,
        help="size of data in GiB to compute on",
    )
    parser.add_argument(
        "--mat-dim",
        type=int,
        default=8192,
        help="dimension of the square weights matrices to multiply against",
    )
    parser.add_argument(
        "--weights-path",
        type=str,
        required=True,
        help="path to weights file",
    )
    parser.add_argument(
        "--load-sep",
        action="store_true",
        help="whether to load each matrix separately (defaults to False)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=20,
        help="number of samples to gather on the workload",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=f"output_{int(time.time())}",
        help="path to output file",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default=f"logs_{int(time.time())}.csv",
        help="path to log file",
    )
    args = parser.parse_args()

    mx.set_default_device(mx.gpu)
    # mx.metal.set_cache_limit(0)
    model = RepresentativeWorkload(args)
    model.execute(args.n_samples)
