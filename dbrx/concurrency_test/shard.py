#!/Users/xiangruike/miniconda3/envs/dbrx_poc/bin/python

from contextlib import AsyncExitStack
from statistics import mean
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

SHARDS = [
    "192.168.1.2:2000",
    "192.168.1.4:4000",
    "192.168.1.5:5000",
    "192.168.1.6:6000",
]

# coroutines to be invoked when the event loop is shutting down
# copied from:
# https://github.com/grpc/grpc/blob/master/examples/python/helloworld/async_greeter_server_with_graceful_shutdown.py
_cleanup_coroutines = []


class Layer:

    def __init__(self, url: str, other_shards: list, block_num: int) -> None:
        self.url = url
        self.other_shards = other_shards
        self.block_num = block_num
        self.buffer = {}
        self.sync_complete = asyncio.Event()

    def reset(self):
        self.buffer = {}
        self.sync_complete.clear()

    async def send(
        self, shard: shard_pb2_grpc.ShardStub, arr_bytes: bytes, arr_map_bytes: bytes
    ):
        await shard.Receive(
            shard_pb2.ShardOuts(
                url=self.url,
                block_num=self.block_num,
                data=arr_bytes,
                arr_map=arr_map_bytes,
            )
        )

    async def calc(self, delay: int, batch_size: int) -> tuple:
        if delay > 0:
            await asyncio.sleep(delay)

        a_bytes = mx_to_bytes(mx.ones((batch_size, 6144), dtype=mx.bfloat16))
        am_bytes = pickle.dumps({})

        tic = time.perf_counter()

        async with asyncio.TaskGroup() as tg:
            for shard in self.other_shards:
                tg.create_task(self.send(shard, a_bytes, am_bytes))

        comm_latency = time.perf_counter() - tic
        tic = time.perf_counter()

        await self.sync_complete.wait()

        sync_latency = time.perf_counter() - tic

        # agg = mx.concatenate([d["expert_outs"] for d in self.buffer.values()], axis=0)
        # mx.eval(agg)

        return comm_latency, sync_latency


class Model:

    def __init__(self, url: str, other_shards: list, n_layers: int) -> None:
        self.n_layers = n_layers
        self.layers = [Layer(url, other_shards, i) for i in range(n_layers)]

    async def start(self, delay: int, batch_size: int):
        comm_latencies = []
        sync_latencies = []

        for layer in self.layers:
            comm_latency, sync_latency = await layer.calc(delay, batch_size)
            layer.reset()

            comm_latencies.append(comm_latency)
            sync_latencies.append(sync_latency)

        print(f"avg comm latency over {self.n_layers} layers: {mean(comm_latencies)}")
        print(f"avg sync latency over {self.n_layers} layers: {mean(sync_latencies)}")


class ShardServicer(shard_pb2_grpc.ShardServicer):

    def __init__(self, url: str) -> None:
        self.url = url
        self.model = None

    def Receive(self, request: shard_pb2.ShardOuts, context):
        layer = self.model.layers[request.block_num]
        layer.buffer[request.url] = {
            "expert_outs": bytes_to_mx(request.data),
            "arr_map": pickle.loads(request.arr_map),
        }

        if len(layer.buffer) == len(SHARDS) - 1:
            layer.sync_complete.set()

        return shard_pb2.Empty()

    async def StartTest(self, request: shard_pb2.Inputs, context):
        async with AsyncExitStack() as es:
            other_shards = []
            for url in SHARDS:
                if url == self.url:
                    continue
                print(url)
                channel = await es.enter_async_context(
                    grpc.aio.insecure_channel(
                        url,
                        options=[
                            ("grpc.max_send_message_length", 9999999),
                            ("grpc.max_receive_message_length", 9999999),
                        ],
                    )
                )
                shard = shard_pb2_grpc.ShardStub(channel)
                other_shards.append(shard)

            self.model = Model(self.url, other_shards, request.n_layers)
            await self.model.start(request.delay, request.batch_size)

        return shard_pb2.Empty()


async def serve(ip: str, port: int):
    server = grpc.aio.server(
        options=[
            ("grpc.max_send_message_length", 9999999),
            ("grpc.max_receive_message_length", 9999999),
        ]
    )
    shard_pb2_grpc.add_ShardServicer_to_server(ShardServicer(f"{ip}:{port}"), server)
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
    parser.add_argument("--ip", type=str)
    parser.add_argument("--port", type=int)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(serve(args.ip, args.port))
    finally:
        loop.run_until_complete(*_cleanup_coroutines)
        loop.close()
