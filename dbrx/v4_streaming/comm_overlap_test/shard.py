#!/Users/xiangruike/miniconda3/envs/dbrx_poc/bin/python

from contextlib import AsyncExitStack
from multiprocessing import connection
from pathlib import Path
from typing import Any
import argparse
import asyncio
import concurrent.futures
import json
import logging
import pickle
import multiprocessing
import time

import grpc
import shard_envoy_pb2
import shard_envoy_pb2_grpc

import mlx.core as mx

from serialization_utils import mx_to_bytes

# coroutines to be invoked when the event loop is shutting down
# copied from:
# https://github.com/grpc/grpc/blob/master/examples/python/helloworld/async_greeter_server_with_graceful_shutdown.py
_cleanup_coroutines = []


class Generator:

    def __init__(self, conn: connection.Connection) -> None:
        self.conn = conn
        x = mx.ones((6144,), dtype=mx.bfloat16)
        self.x_bytes = mx_to_bytes(x)

    def start(self, batch_size: int):
        for _ in range(batch_size):
            time.sleep(0.0009)
            self.conn.send_bytes(self.x_bytes)


def shard_main(conn: connection.Connection) -> None:
    logging.basicConfig(level=logging.INFO)
    generator = Generator(conn)
    logging.info("generator ready")
    batch_size = conn.recv() # confirm that everyone is good to go
    generator.start(batch_size)


class Buffer:

    def __init__(self, n_layers: int, n_oth_shards: int) -> None:
        self.n_layers = n_layers
        self.bin_size = n_oth_shards
        self.data = None
        self.is_full_events = None

    def set_up(self, batch_size: int, for_warming: bool = False) -> None:
        self.data = []
        self.is_full_events = []

        n_bins = batch_size * self.n_layers
        if for_warming:
            n_bins += 1

        for _ in range(n_bins):
            self.data.append([])
            self.is_full_events.append(asyncio.Event())

    def put(self, d: Any, bin_num: int) -> None:
        self.data[bin_num].append(d)
        if len(self.data[bin_num]) == self.bin_size:
            self.is_full_events[bin_num].set()

    def get_data(self, bin_num: int) -> Any:
        return self.data[bin_num]

    def get_is_full_signal(self, bin_num: int) -> asyncio.Event:
        return self.is_full_events[bin_num]


class ShardEnvoyServicer(shard_envoy_pb2_grpc.ShardEnvoyServicer):

    def __init__(
        self, model_path: str, config_filename: str, conn: connection.Connection
    ) -> None:
        self.conn = conn
        self.config = self.get_config(model_path, config_filename)
        self.buffer = Buffer(1, len(self.config["oth_urls"]))

    def get_config(self, model_path: str, config_filename: str) -> dict:
        try:
            with open(Path(model_path) / config_filename, "r") as f:
                tmp = json.load(f)
        except FileNotFoundError:
            logging.error(f"{config_filename} not found in {model_path}")
            raise

        config = {
            "n_layers": tmp["n_layers"],
            "url": tmp["ffn_config"]["shard_url"],
            "oth_urls": [],
        }

        for url in tmp["ffn_config"]["shard_map"]:
            if url == config["url"]:
                continue
            config["oth_urls"].append(url)

        return config

    def broadcast_im_ready(
        self, executor: concurrent.futures.Executor, oth_shards: dict
    ):
        def signal(shard):
            shard.SignalReady(shard_envoy_pb2.Empty())

        fs = []
        for shard in oth_shards.values():
            fut = executor.submit(signal, shard)
            fs.append(fut)
        return fs

    def all_dispatch(
        self,
        bi: int,
        li: int,
        executor: concurrent.futures.Executor,
        oth_shards: dict,
    ) -> None:

        def send(url, shard, data, metadata):
            shard.Receive(shard_envoy_pb2.ShardOuts(data=data, metadata=metadata))
            return url, bi, li

        def when_done(fut):
            logging.info(f"{fut.result()} is done")

        data = self.conn.recv_bytes()
        metadata = pickle.dumps((self.config["url"], bi, li))

        for url, shard in oth_shards.items():
            fut = executor.submit(send, url, shard, data, metadata)
            fut.add_done_callback(when_done)

    def SignalReady(self, request, context):
        logging.info("received signal")
        self.buffer.put(True, -1)
        return shard_envoy_pb2.Empty()

    def Receive(self, request: shard_envoy_pb2.ShardOuts, context):
        logging.info(f"started receiving")

        url, bi, li = pickle.loads(request.metadata)
        self.buffer.put((request.data, request.metadata), bi * li)

        logging.info(f"finished receiving")
        return shard_envoy_pb2.Empty()

    async def Start(self, request: shard_envoy_pb2.TestIns, context):
        self.buffer.set_up(request.batch_size, for_warming=True)
        async with AsyncExitStack() as es:
            oth_shards = {}
            for url in self.config["oth_urls"]:
                channel = await es.enter_async_context(
                    grpc.aio.insecure_channel(
                        url,
                        options=[
                            ("grpc.max_send_message_length", -1),
                            ("grpc.max_receive_message_length", -1),
                        ],
                    )
                )
                shard = shard_envoy_pb2_grpc.ShardEnvoyStub(channel)
                oth_shards[url] = shard

            with concurrent.futures.ThreadPoolExecutor() as executor:

                concurrent.futures.wait(self.broadcast_im_ready(executor, oth_shards))
                await self.buffer.get_is_full_signal(-1).wait()
                logging.info("everyone is ready")
                self.conn.send(request.batch_size) # signal generator to start working

                for bi in range(request.batch_size):
                    self.all_dispatch(bi, 0, executor, oth_shards)


async def serve(
    port: int, model_path: str, config_filename: str, conn: connection.Connection
):
    server = grpc.aio.server(
        options=[
            ("grpc.max_send_message_length", -1),
            ("grpc.max_receive_message_length", -1),
        ]
    )
    servicer = ShardEnvoyServicer(model_path, config_filename, conn)
    shard_envoy_pb2_grpc.add_ShardEnvoyServicer_to_server(servicer, server)
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


def envoy_main(
    port: int, model_path: str, config_filename: str, conn: connection.Connection
):
    logging.basicConfig(level=logging.INFO)
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(serve(port, model_path, config_filename, conn))
    finally:
        loop.run_until_complete(*_cleanup_coroutines)
        loop.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int)
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--config-filename", type=str)
    args = parser.parse_args()

    envoy_conn, shard_conn = multiprocessing.Pipe()

    envoy_p = multiprocessing.Process(
        target=envoy_main,
        args=(args.port, args.model_path, args.config_filename, envoy_conn),
    )
    shard_p = multiprocessing.Process(target=shard_main, args=(shard_conn,))

    envoy_p.start()
    shard_p.start()

    envoy_p.join()
    shard_p.join()
