import mlx.core as mx


def simple_computations(a, b, d1, d2):
    # copied from:
    # https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html

    # compute dense, more suitable for GPU
    # dimension: 4096 * 4
    # 4096 * 4 / 1024 = 16 KiB
    x = mx.matmul(a, b, stream=d1)
    mx.eval(x)
    for _ in range(500):
        # less dense, faster on CPU
        b = mx.exp(b, stream=d2)
    mx.eval(b)


if __name__ == "__main__":
    # default return dtype of mx.random.uniform is float32
    # (4096 * 512) / (1024 * 1024) = 2 GiB
    a = mx.random.uniform(shape=(4096, 512))
    # (512 * 4) / 1024 = 2 KiB
    b = mx.random.uniform(shape=(512, 4))

    simple_computations(a, b, mx.gpu, mx.cpu)
