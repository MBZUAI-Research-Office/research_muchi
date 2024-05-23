from multiprocessing import connection
import multiprocessing
import time
import statistics

import mlx.core as mx

from serialization_utils import mx_to_bytes, bytes_to_mx

def comm_test0(conn: connection.Connection, n: int):
    for _ in range(n):
        a_bytes = conn.recv_bytes()
        conn.send_bytes(a_bytes)

    conn.close()

def shard_test0(conn: connection.Connection, n: int):
    # pure overhead of isolating comm module using multiprocessing and pipe

    arr = mx.random.uniform(-1, 1, (6144,), dtype=mx.bfloat16)
    mx.eval(arr)
    out_bytes = mx_to_bytes(arr)

    latencies = []
    for _ in range(n):
        tic = time.perf_counter_ns()

        conn.send_bytes(out_bytes)
        in_bytes = conn.recv_bytes()

        latencies.append(time.perf_counter_ns() - tic)

    recovered = bytes_to_mx(in_bytes)
    assert mx.array_equal(arr, recovered).item

    print(f"avg latency: {statistics.mean(latencies) / 1000} microsecond(s)")
    conn.close()

def shard_test1(conn: connection.Connection, n: int):
    # includes serialization

    arr = mx.random.uniform(-1, 1, (6144,), dtype=mx.bfloat16)
    mx.eval(arr)

    latencies = []
    for _ in range(n):
        tic = time.perf_counter_ns()

        out_bytes = mx_to_bytes(arr)
        conn.send_bytes(out_bytes)

        in_bytes = conn.recv_bytes()
        recovered = bytes_to_mx(in_bytes)
        mx.eval(recovered)

        latencies.append(time.perf_counter_ns() - tic)

    assert mx.array_equal(arr, recovered).item

    print(f"avg latency: {statistics.mean(latencies) / 1000} microsecond(s)")
    conn.close()

def main(): 
    n = 100_000
    comm_conn, shard_conn = multiprocessing.Pipe()

    comm_p = multiprocessing.Process(target=comm_test0, args=(comm_conn, n)) 
    shard_p = multiprocessing.Process(target=shard_test0, args=(shard_conn, n)) 
    # shard_p = multiprocessing.Process(target=shard_test1, args=(shard_conn, n)) 

    comm_p.start() 
    shard_p.start()

    comm_p.join()
    shard_p.join()


def ser_latency_test0():
    arr = mx.random.uniform(-1, 1, (6144,), dtype=mx.bfloat16)
    mx.eval(arr)

    latencies = []
    for _ in range(10000):
        tic = time.perf_counter_ns()
        a_bytes = mx_to_bytes(arr)
        mx.eval(bytes_to_mx(a_bytes))
        latencies.append(time.perf_counter_ns() - tic)
    
    print(f"AVG ser + de-ser latency: {statistics.mean(latencies) / 1000} microsecond(s)")

if __name__ == "__main__":
    main()
