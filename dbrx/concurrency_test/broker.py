#!/Users/xiangruike/miniconda3/envs/dbrx_poc/bin/python

from contextlib import AsyncExitStack
import asyncio

import grpc
import shard_pb2
import shard_pb2_grpc


async def StartTest(shard: shard_pb2_grpc.ShardStub):
    await shard.StartTest(shard_pb2.Empty())


async def test0():
    async with AsyncExitStack() as es:
        shards = []
        for url in ["192.168.1.6:6000", "192.168.1.6:6001", "192.168.1.6:6002"]:
            channel = await es.enter_async_context(grpc.aio.insecure_channel(url))
            shard = shard_pb2_grpc.ShardStub(channel)
            shards.append(shard)

        async with asyncio.TaskGroup() as tg:
            for shard in shards:
                tg.create_task(StartTest(shard))


if __name__ == "__main__":
    print("test started")
    asyncio.run(test0())
    print("test finished")
