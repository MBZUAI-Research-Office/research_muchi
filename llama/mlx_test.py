#!/Users/xiangruike/miniconda3/envs/custom_mlx_lm/bin/python

import argparse
import time

import mlx.core as mx


class RepresentativeWorkload:

    DEFAULT_INPUT_M = 32
    DEFAULT_INPUT_N = 8192

    # mx.random.uniform() is used to produce the weights, of which
    # the default return dtype is float32. Therefore, the size of
    # each weights matrix is:
    # (16384 * 16384 * 4) / (1024 * 1024 * 1024) = 1 GiB
    DEFAULT_WEIGHTS_M = 8192
    DEFAULT_WEIGHTS_N = 8192
    DEFAULT_WEIGHTS_SIZE = 0.25  # in GiB

    def __init__(self, total_weights_size: int, load_by_layer: bool) -> None:
        """Encapsulates a simple yet memory demanding workload

        Args:
            total_weights_size (int): size of weights in GiB to load into unified memory.
        """
        assert total_weights_size % self.DEFAULT_WEIGHTS_SIZE == 0

        mx.set_default_device(mx.gpu)
        self.weights = self.load_weights(total_weights_size, load_by_layer)

    def load_weights(self, total_weights_size: int, load_by_layer: bool):
        tic = time.perf_counter()

        num_layers = int(total_weights_size / self.DEFAULT_WEIGHTS_SIZE)
        if load_by_layer:
            weights = []
            for _ in range(num_layers):
                weights.append(
                    mx.random.uniform(
                        shape=(
                            self.DEFAULT_WEIGHTS_M,
                            self.DEFAULT_WEIGHTS_N,
                        )
                    )
                )
        else:
            weights = mx.random.uniform(
                shape=(
                    num_layers,
                    self.DEFAULT_WEIGHTS_M,
                    self.DEFAULT_WEIGHTS_N,
                )
            )
        mx.eval(weights)

        print(f"LOAD LATENCY: {time.perf_counter() - tic} seconds", flush=True)
        print("=" * 10, flush=True)
        return weights

    def generate_token(self) -> None:
        token = mx.random.uniform(shape=(self.DEFAULT_INPUT_M, self.DEFAULT_INPUT_N))
        for layer in self.weights:
            token = mx.matmul(token, layer)
        mx.eval(token)

    def generate(self, n_tokens: int) -> None:
        print("Generation Started", flush=True)
        print("=" * 10, flush=True)

        total_latency = 0
        for i in range(n_tokens):
            tic = time.perf_counter()

            self.generate_token()

            toc = time.perf_counter()
            latency = toc - tic
            total_latency += latency
            print(f"#{i} token generated in {latency} seconds", flush=True)

        print("=" * 10)
        print(f"AVERAGE LATENCY: {total_latency / n_tokens} seconds")
        print(f"TOTAL LATENCY: {total_latency} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="runs llama 2 with mlx_lm")
    parser.add_argument(
        "--total-weights-size",
        "-t",
        type=int,
        help="total weights size in GiB",
    )
    parser.add_argument(
        "--n-tokens",
        "-n",
        type=int,
        help="number of tokens to generate",
    )
    parser.add_argument(
        "--load-by-layer",
        "-l",
        action="store_true",
        help="whether to load each layer's weights separately",
    )
    args = parser.parse_args()

    model = RepresentativeWorkload(
        total_weights_size=args.total_weights_size, load_by_layer=args.load_by_layer
    )
    model.generate(args.n_tokens)
