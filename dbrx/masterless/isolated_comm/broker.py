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


DEFAULT_PROMPT = "hello"
DEFAULT_MAX_TOKENS = 100


def get_shard_urls(config_path: Path) -> list:
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        logging.error(f"{config_path} not found")
        raise

    return list(config["ffn_config"]["shard_map"].keys())


async def call(
    shard: shard_envoy_pb2_grpc.ShardEnvoyStub, prompt: str, max_tokens: int
) -> shard_envoy_pb2.UsrOuts:
    return await shard.Start(shard_envoy_pb2.UsrIns(prompt=prompt, max_tokens=max_tokens))


async def start(config_path: str, prompt: str, max_tokens: int) -> None:
    print("INFERENCE STARTED:")
    print(f"PROMPT: {prompt}")

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
                inference_tasks.append(tg.create_task(call(shard, prompt, max_tokens)))

        output = inference_tasks[0].result()

    if output.gen_t_cnt == 0:
        print("No tokens generated for this prompt")
        return

    print("RESPONSE:")
    print(output.response)
    print("PROMPT EVALUATION:")
    print(f"token count: {output.prompt_t_cnt}")
    print(f"total time in sec(s): {output.prompt_time:.3f}")
    print(f"throughput: {(output.prompt_t_cnt / output.prompt_time):.3f} t/s")
    print("TOKEN GENERATION:")
    print(f"token count: {output.gen_t_cnt - 1}")
    print(f"total time in sec(s): {output.gen_time:.3f}")
    print(f"throughput: {((output.gen_t_cnt - 1) / output.gen_time):.3f} t/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str)
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Message to be processed by the model",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum number of tokens to generate",
    )
    args = parser.parse_args()

    logging.basicConfig()
    asyncio.run(start(args.config_path, args.prompt, args.max_tokens))
