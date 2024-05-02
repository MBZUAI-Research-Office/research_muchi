import io
import mlx.core as mx


def mx_to_bytes(arr: mx.array) -> bytes:
    # convert mx.array to bytes
    arr_bytes = None
    with io.BytesIO() as buffer:
        mx.savez(buffer, arr=arr)
        arr_bytes = buffer.getvalue()
    assert arr_bytes is not None
    return arr_bytes


def bytes_to_mx(a_bytes: bytes) -> mx.array:
    # convert bytes to mx.array
    arr = None
    with io.BytesIO(a_bytes) as buffer:
        buffer.name = "xxx.npz"  # hack!
        arr = mx.load(buffer)
    assert arr is not None
    return arr["arr"]


# def test_0():
#     a = mx.random.normal((1, 6144), dtype=mx.bfloat16)
#     print(f"{a=}")

#     a_bytes = mx_to_bytes(a)
#     # print(a_bytes)
#     import sys

#     print(sys.getsizeof(a_bytes))

#     b = bytes_to_mx(a_bytes)
#     print(f"{b=}")

#     print(f"{mx.equal(a,b)}")


# if __name__ == "__main__":
#     test_0()
