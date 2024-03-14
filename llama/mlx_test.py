#!/Users/xiangruike/miniconda3/envs/custom_mlx_lm/bin/python

import argparse
import time

import mlx.core as mx


class RepresentativeWorkload:

    # 25 matrices/layers, input_m = 32
    DEFAULT_NUM_MULS = 858993459200

    def __init__(
        self, data_size: int, mat_dim: int, load_sep: bool, log_path: str
    ) -> None:
        weights_size = (4 * mat_dim**2) / 1024**3
        assert data_size % weights_size == 0
        self.mat_dim = mat_dim
        self.load_sep = load_sep
        self.log_path = log_path
        self.num_weights = int(data_size / weights_size)
        self.input_m = int(self.DEFAULT_NUM_MULS / (self.num_weights * mat_dim**2))
        self.latencies = {}
        self.weights = self.load_weights(mat_dim)

    def load_weights(self, mat_dim: int):
        tic = time.perf_counter()

        if self.load_sep:
            weights = []
            for _ in range(self.num_weights):
                weights.append(mx.random.uniform(shape=(mat_dim, mat_dim)))
        else:
            weights = mx.random.uniform(shape=(self.num_weights, mat_dim, mat_dim))
        mx.eval(weights)

        self.latencies["load_latency"] = time.perf_counter() - tic
        print(f"LOAD LATENCY: {self.latencies['load_latency']} seconds", flush=True)

        return weights

    def execute_step(self) -> float:
        token = mx.random.uniform(shape=(self.input_m, self.mat_dim))

        tic = time.perf_counter()

        for layer in self.weights:
            token = mx.matmul(token, layer)
        mx.eval(token)

        toc = time.perf_counter()
        return toc - tic

    def execute(self, n_samples: int) -> None:
        total_latency = 0
        for _ in range(n_samples):
            latency = self.execute_step()
            total_latency += latency

        self.latencies["avg_compute_latency"] = total_latency / n_samples
        self.latencies["total_compute_latency"] = total_latency
        print(f"AVERAGE LATENCY: {self.latencies['avg_compute_latency']} seconds")
        print(f"TOTAL LATENCY: {total_latency} seconds")

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
        help="size of data in GiB to compute on",
    )
    parser.add_argument(
        "--mat-dim",
        type=int,
        help="dimension of the square weights matrices to multiply against",
    )
    parser.add_argument(
        "--load-sep",
        action="store_true",
        help="whether to load each matrix separately (defaults to False)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        help="number of samples to gather on the workload",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        help="path to log file",
    )
    args = parser.parse_args()

    mx.set_default_device(mx.gpu)
    model = RepresentativeWorkload(
        data_size=args.data_size,
        mat_dim=args.mat_dim,
        load_sep=args.load_sep,
        log_path=args.log_path,
    )
    model.execute(args.n_samples)
