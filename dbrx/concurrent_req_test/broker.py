#!/Users/xiangruike/miniconda3/envs/dbrx_poc/bin/python

import asyncio
import logging
import time

import grpc
import concurrent_shard_pb2
import concurrent_shard_pb2_grpc


async def call(shard: concurrent_shard_pb2_grpc.ConcurrentShardStub, secs: int) -> None:
    tic = time.perf_counter()
    await shard.Start(concurrent_shard_pb2.TestIns(secs=secs))
    logging.info(f"expected {secs}, got {time.perf_counter() - tic} sec(s)")


async def start() -> None:
    async with grpc.aio.insecure_channel("192.168.1.6:6000") as channel:
        shard = concurrent_shard_pb2_grpc.ConcurrentShardStub(channel)
        async with asyncio.TaskGroup() as tg:
            for i in range(1, 6):
                tg.create_task(call(shard, i))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(start())
