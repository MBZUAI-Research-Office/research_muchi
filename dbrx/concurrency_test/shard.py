#!/Users/xiangruike/miniconda3/envs/dbrx_poc/bin/python

from concurrent import futures
from contextlib import AsyncExitStack
import argparse
import asyncio
import logging
import pickle
import time

import grpc
import shard_pb2
import shard_pb2_grpc

import mlx.core as mx

from serialization_utils import mx_to_bytes, bytes_to_mx

SHARDS = ["192.168.1.6:6000", "192.168.1.6:6001", "192.168.1.6:6002"]

# coroutines to be invoked when the event loop is shutting down
# copied from:
# https://github.com/grpc/grpc/blob/master/examples/python/helloworld/async_greeter_server_with_graceful_shutdown.py
_cleanup_coroutines = []


class Layer:

    def __init__(self, url: str, other_shards: list, buffer: dict) -> None:
        self.url = url
        self.other_shards = other_shards
        self.buffer = buffer

    async def send(
        self, shard: shard_pb2_grpc.ShardStub, arr_bytes: bytes, arr_map_bytes: bytes
    ):
        await shard.Receive(
            shard_pb2.ShardOuts(
                url=self.url,
                data=arr_bytes,
                arr_map=arr_map_bytes,
            )
        )

    async def calc(self, delay: int, sync_complete: asyncio.Event) -> None:
        print(f"started dispatching at {time.time()}", flush=True)

        if delay > 0:
            await asyncio.sleep(delay)

        a_bytes = mx_to_bytes(mx.random.uniform(-1, 1, (1, 6144), dtype=mx.bfloat16))
        am_bytes = pickle.dumps({})
        async with asyncio.TaskGroup() as tg:
            for shard in self.other_shards:
                tg.create_task(self.send(shard, a_bytes, am_bytes))

        print(f"ended dispatching at {time.time()}", flush=True)

        await sync_complete.wait()

        print(f"started reducing at {time.time()}", flush=True)

        agg = mx.concatenate([d["expert_outs"] for d in self.buffer.values()], axis=0)
        mx.eval(agg)

        print(f"ended reducing at {time.time()}", flush=True)


class Model:

    def __init__(self, url: str, other_shards: list, n_layers: int, delay: int) -> None:
        self.delay = delay
        self.sync_complete = asyncio.Event()
        self.buffer = {}
        self.layers = [Layer(url, other_shards, self.buffer) for _ in range(n_layers)]

    def reset(self):
        self.sync_complete.clear()
        self.buffer = {}

    async def start(self):
        tic = time.perf_counter()

        for layer in self.layers:
            await layer.calc(self.delay, self.sync_complete)
            self.reset()

        print(f"shard finished in {time.perf_counter() - tic} seconds", flush=True)


class ShardServicer(shard_pb2_grpc.ShardServicer):

    def __init__(self, url: str, n_layers: int, delay: int) -> None:
        self.url = url
        self.n_layers = n_layers
        self.delay = delay
        self.model = None

    def Receive(self, request: shard_pb2.ShardOuts, context):
        print(f"started receiving from {request.url} at {time.time()}", flush=True)

        self.model.buffer[request.url] = {
            "expert_outs": bytes_to_mx(request.data),
            "arr_map": pickle.loads(request.arr_map),
        }

        if len(self.model.buffer) == len(SHARDS) - 1:
            self.model.sync_complete.set()

        print(f"ended receiving from {request.url} at {time.time()}", flush=True)
        return shard_pb2.Empty()

    async def StartTest(self, request, context):
        async with AsyncExitStack() as es:
            other_shards = []
            for url in SHARDS:
                if url == self.url:
                    continue
                channel = await es.enter_async_context(grpc.aio.insecure_channel(url))
                shard = shard_pb2_grpc.ShardStub(channel)
                other_shards.append(shard)

            self.model = Model(self.url, other_shards, self.n_layers, self.delay)
            await self.model.start()

        return shard_pb2.Empty()


async def serve(port: int, n_layers: int, delay: int):
    server = grpc.aio.server()
    servicer = ShardServicer(f"192.168.1.6:{port}", n_layers, delay)
    shard_pb2_grpc.add_ShardServicer_to_server(servicer, server)
    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)
    await server.start()
    logging.info(f"server started, listening on {listen_addr}")

    # copied from:
    # https://github.com/grpc/grpc/blob/master/examples/python/helloworld/async_greeter_server_with_graceful_shutdown.py
    async def server_graceful_shutdown():
        logging.info("Starting graceful shutdown...")
        # Shuts down the server with 3 seconds of grace period. During the
        # grace period, the server won't accept new connections and allow
        # existing RPCs to continue within the grace period.
        await server.stop(3)

    _cleanup_coroutines.append(server_graceful_shutdown())
    await server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int)
    parser.add_argument("--n-layers", type=int)
    parser.add_argument("--delay", type=int)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(serve(args.port, args.n_layers, args.delay))
    finally:
        loop.run_until_complete(*_cleanup_coroutines)
        loop.close()
