#!/Users/xiangruike/miniconda3/envs/custom_mlx_lm/bin/python

import argparse
import time

import mlx.core as mx

# in GB
DEFAULT_DATA_SIZE = 10
# loads all weights in a giant array by default to optimize memory access
DEFAULT_N_BLOCKS = 1
DEFAULT_MAT_DIM = 5000


class RepresentativeWorkload:

    def __init__(
        self,
        data_size: int,
        n_blocks: int,
        mat_dim: int,
    ) -> None:
        # square matrix with 4 bytes (float32) cells
        mat_size = 4 * (mat_dim**2)
        data_size *= 10**9
        assert data_size % (n_blocks * mat_size) == 0

        self.n_mats_per_block = int(data_size / (n_blocks * mat_size))
        self.data_size = data_size
        self.n_blocks = n_blocks
        self.mat_dim = mat_dim
        self.data = self.load_data()

    def load_data(self):
        if self.n_blocks > 1:
            print(
                "------ "
                + f"loading {self.n_blocks} blocks of data sequentially as individual arrays "
                + f"{int(self.data_size / self.n_blocks)} bytes at a time "
                + "------",
                flush=True,
            )

        tic = time.perf_counter()

        data = []
        for _ in range(self.n_blocks):
            data.append(
                mx.random.uniform(
                    shape=(
                        self.n_mats_per_block,
                        self.mat_dim,
                        self.mat_dim,
                    )
                )
            )
        mx.eval(data)

        print(f"data loading clock time: {time.perf_counter() - tic} seconds")

        return data

    def execute(self) -> None:
        tic = time.perf_counter()

        res = mx.identity(self.mat_dim)
        for i in range(self.n_blocks * self.n_mats_per_block):
            offset = i // self.n_mats_per_block
            number = i % self.n_mats_per_block
            res = mx.matmul(res, self.data[offset][number])
        mx.eval(res)

        print(f"computation clock time: {time.perf_counter() - tic} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="a simple yet memory demanding workload"
    )
    parser.add_argument(
        "--data-size",
        type=int,
        default=DEFAULT_DATA_SIZE,
        help=f"total data size in GB, (defaults to {DEFAULT_DATA_SIZE})",
    )
    parser.add_argument(
        "--n-blocks",
        type=int,
        default=DEFAULT_N_BLOCKS,
        help="load data in the specified number of separate blocks, "
        + f"(defaults to {DEFAULT_N_BLOCKS})",
    )
    parser.add_argument(
        "--mat-dim",
        type=int,
        default=DEFAULT_MAT_DIM,
        help="size of the square matrices to perform multiplications on, "
        + f"(defaults to {DEFAULT_MAT_DIM})",
    )
    args = parser.parse_args()

    mx.set_default_device(mx.gpu)
    job = RepresentativeWorkload(
        args.data_size,
        args.n_blocks,
        args.mat_dim,
    )
    job.execute()
