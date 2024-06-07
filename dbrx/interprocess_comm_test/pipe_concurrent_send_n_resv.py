from multiprocessing import connection
import concurrent.futures
import multiprocessing
import time

import mlx.core as mx

from serialization_utils import mx_to_bytes, bytes_to_mx

def envoy_test0(envoy_send: connection.Connection, shard_resv: connection.Connection, n: int):

    for _ in range(n):
        a_bytes = shard_resv.recv_bytes()
        time.sleep(0.001)
        envoy_send.send_bytes(a_bytes)

    envoy_send.close()
    shard_resv.close()

def shard_test0(envoy_resv: connection.Connection, shard_send: connection.Connection, n: int):
    # includes serialization

    arr = mx.random.uniform(-1, 1, (6144,), dtype=mx.bfloat16)
    mx.eval(arr)

    tic = time.perf_counter_ns()

    for _ in range(n):
        time.sleep(0.001)
        shard_send.send_bytes(mx_to_bytes(arr))
        a_bytes = envoy_resv.recv_bytes()
        mx.eval(bytes_to_mx(a_bytes))

    print(f"avg latency: {(time.perf_counter_ns() - tic) / (n * 1000)} microsecond(s)")

    envoy_resv.close()
    shard_send.close()

def shard_test1(envoy_resv: connection.Connection, shard_send: connection.Connection, n: int):
    # includes serialization

    def compute(arr):
        for _ in range(n):
            time.sleep(0.001)
            shard_send.send_bytes(mx_to_bytes(arr))

    def communication():
        for _ in range(n):
            a_bytes = envoy_resv.recv_bytes()
            mx.eval(bytes_to_mx(a_bytes))

    arr = mx.random.uniform(-1, 1, (6144,), dtype=mx.bfloat16)
    mx.eval(arr)

    tic = time.perf_counter_ns()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(compute, arr)
        executor.submit(communication)

    print(f"avg latency: {(time.perf_counter_ns() - tic) / (n * 1000)} microsecond(s)")

    envoy_resv.close()
    shard_send.close()

def main(): 
    n = 10_000
    envoy_resv, envoy_send = multiprocessing.Pipe(duplex=False)
    shard_resv, shard_send = multiprocessing.Pipe(duplex=False)

    envoy_p = multiprocessing.Process(target=envoy_test0, args=(envoy_send, shard_resv, n))
    shard_p = multiprocessing.Process(target=shard_test0, args=(envoy_resv, shard_send, n))
    # shard_p = multiprocessing.Process(target=shard_test1, args=(envoy_resv, shard_send, n))

    envoy_p.start() 
    shard_p.start()

    envoy_p.join()
    shard_p.join()

if __name__ == "__main__":
    main()
