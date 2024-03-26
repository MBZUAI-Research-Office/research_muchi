#!/Users/xiangruike/miniconda3/envs/mlx_perf/bin/python

import argparse

import numpy as np

def compare_outputs(output_0: np.array, output_1: np.array) -> bool:
    arr_0 = np.load(output_0)
    arr_1 = np.load(output_1)
    return np.array_equal(arr_0, arr_1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="a simple yet memory demanding workload"
    )
    parser.add_argument(
        "--output-0",
        type=str,
        help="path to output_0",
    )
    parser.add_argument(
        "--output-1",
        type=str,
        help="path to output_1",
    )
    args = parser.parse_args()
    are_equal = compare_outputs(args.output_0, args.output_1)
    print(f"provided outputs are {'' if are_equal else 'not '}equal")
    print("=" * 20)
