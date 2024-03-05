#!/Users/xiangruike/miniconda3/envs/custom_mlx_lm/bin/python

import argparse
import time

import mlx.core as mx


class RepresentativeWorkload:

    # mx.random.uniform() is used to produce the weights, of which
    # the default return dtype is float32. Therefore, the size of
    # each weights matrix is:
    # (32768 * 32768 * 4) / (1024 * 1024 * 1024) = 4 GiB
    WEIGHTS_M = 32768
    WEIGHTS_N = 32768
    WEIGHTS_SIZE = 4  # in GiB

    def __init__(
        self,
        total_weights_size: int,
        n_load_blocks: int,
        n_matmuls_per_layer: int,
        n_tokens: int,
    ) -> None:
        """Encapsulates a simple yet memory demanding workload

        Args:
            total_weights_size (int): size of weights in GiB to load into unified memory.
        """
        self.total_weights_size = total_weights_size
        self.n_load_blocks = n_load_blocks
        self.n_matmuls_per_layer = n_matmuls_per_layer
        self.n_tokens = n_tokens
        self.num_layers = int(self.total_weights_size / self.WEIGHTS_SIZE)

        assert self.total_weights_size % self.n_load_blocks == 0
        self.weights_block_size = int(total_weights_size / n_load_blocks)

        assert self.weights_block_size % self.WEIGHTS_SIZE == 0
        self.n_weights_per_block = int(self.weights_block_size / self.WEIGHTS_SIZE)

        self.weights = self.load_weights(n_load_blocks)

    def load_weights(self):
        if self.n_load_blocks > 1:
            print(
                "\n------ "
                + f"loading {self.n_load_blocks} blocks of weights sequentially as individual arrays "
                + f"{self.weights_block_size} GiB at a time "
                + "------",
                flush=True,
            )

        weights = []

        for _ in range(self.n_load_blocks):
            weights.append(
                mx.random.uniform(
                    shape=(
                        self.n_weights_per_block,
                        self.WEIGHTS_M,
                        self.WEIGHTS_N,
                    )
                )
            )

        # mlx has lazy evaluation: forcing load to happen here
        mx.eval(weights)

        return weights

    def generate_token(self) -> None:
        token = None
        for i in range(self.num_layers):
            offset = i // self.n_weights_per_block
            number = i % self.n_weights_per_block
            layer_weights = self.weights[offset][number]
            remaining_matmuls = self.n_matmuls_per_layer

            if token is None:
                # 1. only for the first time
                # 2. eliminates the need for an input
                # 3. defines a configurable unit of computation per layer: matmul
                token = mx.matmul(layer_weights, layer_weights)
                remaining_matmuls -= 1

            for _ in range(remaining_matmuls):
                mx.matmul(token, layer_weights)
        mx.eval(token)

    def generate(self) -> None:
        print("Generation Started", flush=True)
        print("=" * 10, flush=True)

        total_latency = 0
        for i in range(self.n_tokens):
            tic = time.perf_counter()

            self.generate_token()

            toc = time.perf_counter()
            latency = toc - tic
            total_latency += latency
            print(f"#{i} token generated in {latency} seconds", flush=True)

        print("=" * 10)
        print(f"AVG t/s: {self.n_tokens / total_latency}")
        print(f"TOTAL LATENCY: {total_latency} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="runs llama 2 with mlx_lm")
    parser.add_argument(
        "--total-weights-size",
        "-w",
        type=int,
        help="total weights size in GiB",
    )
    parser.add_argument(
        "--n-load-blocks",
        "-b",
        type=int,
        default=1,
        help="load weights in the specified number separate blocks",
    )
    parser.add_argument(
        "--matmuls-per-layer",
        "-m",
        type=int,
        default=5,  # Q, K, V, Q * K_T, prev_results * V
        help="number of matrix multiplications to perform per layer",
    )
    parser.add_argument(
        "--n-tokens",
        "-n",
        type=int,
        help="number of tokens to generate",
    )
    args = parser.parse_args()

    mx.set_default_device(mx.gpu)
    model = RepresentativeWorkload(
        args.total_weights_size,
        args.n_load_blocks,
        args.matmuls_per_layer,
        args.n_tokens,
    )
    model.generate()
