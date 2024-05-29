#!/Users/xiangruike/miniconda3/envs/dbrx_poc/bin/python

from contextlib import AsyncExitStack
from pathlib import Path
import argparse
import asyncio
import json
import logging

import grpc
import shard_envoy_pb2
import shard_envoy_pb2_grpc


def get_shard_urls(config_path: Path) -> list:
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        logging.error(f"{config_path} not found")
        raise

    return list(config["ffn_config"]["shard_map"].keys())


async def call(shard: shard_envoy_pb2_grpc.ShardEnvoyStub, batch_size: int) -> None:
    await shard.Start(shard_envoy_pb2.TestIns(batch_size=batch_size))


async def start(config_path: str, batch_size: int) -> None:
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
            shard = shard_envoy_pb2_grpc.ShardEnvoyStub(channel)
            shards.append(shard)

        async with asyncio.TaskGroup() as tg:
            inference_tasks = []
            for shard in shards:
                inference_tasks.append(tg.create_task(call(shard, batch_size)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()

    logging.basicConfig()
    asyncio.run(start(args.config_path, args.batch_size))
