#!/Users/xiangruike/miniconda3/envs/dbrx_poc/bin/python

from contextlib import AsyncExitStack
import argparse
import asyncio

import grpc
import shard_pb2
import shard_pb2_grpc


async def StartTest(
    shard: shard_pb2_grpc.ShardStub, n_layers: int, delay: int, batch_size: int
):
    await shard.StartTest(
        shard_pb2.Inputs(n_layers=n_layers, delay=delay, batch_size=batch_size)
    )


async def test0(n_layers: int, delay: int, batch_size: int):
    async with AsyncExitStack() as es:
        shards = []
        for url in [
            "192.168.1.2:2000",
            "192.168.1.4:4000",
            "192.168.1.5:5000",
            "192.168.1.6:6000",
        ]:
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
            shards.append(shard)

        async with asyncio.TaskGroup() as tg:
            for shard in shards:
                tg.create_task(StartTest(shard, n_layers, delay, batch_size))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-layers", type=int)
    parser.add_argument("--delay", type=int)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()

    print("test started")
    asyncio.run(test0(args.n_layers, args.delay, args.batch_size))
    print("test finished")
