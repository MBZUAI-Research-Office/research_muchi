#!/Users/xiangruike/miniconda3/envs/dbrx_poc/bin/python

from contextlib import AsyncExitStack
from pathlib import Path
import argparse
import asyncio
import json
import logging
import time

import grpc
import shard_pb2
import shard_pb2_grpc


def get_shard_urls(config_path: Path) -> list:
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        logging.error(f"{config_path} not found")
        raise

    return list(config["ffn_config"]["shard_map"].keys())


async def call(shard: shard_pb2_grpc.ShardStub) -> shard_pb2.Outputs:
    await shard.Start(shard_pb2.Inputs())


async def start(config_path: str) -> None:
    shard_urls = get_shard_urls(Path(config_path))
    async with AsyncExitStack() as es:
        shards = []
        for url in shard_urls:
            channel = await es.enter_async_context(
                grpc.aio.insecure_channel(
                    url,
                    options=[
                        ("grpc.max_send_message_length", -1),
                        ("grpc.max_receive_message_length", -1),
                    ],
                )
            )
            shard = shard_pb2_grpc.ShardStub(channel)
            shards.append(shard)

        async with asyncio.TaskGroup() as tg:
            inference_tasks = []
            for shard in shards:
                inference_tasks.append(tg.create_task(call(shard)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str)
    args = parser.parse_args()

    logging.basicConfig()
    print("test started", flush=True)
    tic = time.perf_counter()
    asyncio.run(start(args.config_path))
    print(f"test finished in {(time.perf_counter() - tic):.3f} sec(s)", flush=True)
