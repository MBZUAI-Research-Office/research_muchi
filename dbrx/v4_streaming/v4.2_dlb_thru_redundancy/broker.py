#!/Users/xiangruike/miniconda3/envs/dbrx_enhanced/bin/python

from contextlib import AsyncExitStack
from pathlib import Path
import argparse
import asyncio
import json
import logging
import pprint
import statistics

import grpc
import shard_envoy_pb2
import shard_envoy_pb2_grpc

STATS = {"moe_lat": [], "comm_lat": [], "misc_lat": [], "experts_act": [], "prompt_eval_tp": [], "token_gen_tp": []}
DEFAULT_MAX_TOKENS = 100


def get_json(file_path: Path) -> list:
    try:
        with open(file_path, "r") as f:
            res = json.load(f)
    except FileNotFoundError:
        logging.error(f"{file_path} not found")
        raise

    return res


async def call(
    shard: shard_envoy_pb2_grpc.ShardEnvoyStub, prompt: str, max_tokens: int
) -> shard_envoy_pb2.UsrOuts:
    return await shard.Generate(
        shard_envoy_pb2.UsrIns(prompt=prompt, max_tokens=max_tokens)
    )


async def make_inference_requests(
    shards: list[shard_envoy_pb2_grpc.ShardEnvoyStub], prompt: str, max_tokens: int
):
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
    prompt_eval_tp = output.prompt_t_cnt / output.prompt_time
    print(f"throughput: {prompt_eval_tp:.3f} t/s")
    print("TOKEN GENERATION:")
    print(f"token count: {output.gen_t_cnt - 1}")
    print(f"total time in sec(s): {output.gen_time:.3f}")
    token_gen_tp = (output.gen_t_cnt - 1) / output.gen_time
    print(f"throughput: {token_gen_tp:.3f} t/s")

    if output.gen_t_cnt >= max_tokens * 0.85:
        avg_misc_lat = (1000 / token_gen_tp / 40) - output.avg_moe_lat - output.avg_comm_lat
        STATS["moe_lat"].append(output.avg_moe_lat)
        STATS["comm_lat"].append(output.avg_comm_lat)
        STATS["misc_lat"].append(avg_misc_lat)
        STATS["experts_act"].append(output.avg_experts_act)
        STATS["prompt_eval_tp"].append(prompt_eval_tp)
        STATS["token_gen_tp"].append(token_gen_tp)
        return True

    return False


async def start(
    config_path: str, prompt: str, prompt_path: str, n_samples: int, max_tokens: int
) -> None:
    assert max_tokens > 0

    print("INFERENCE STARTED:")

    shard_urls = get_json(Path(config_path))["shard_urls"]
    prompts = get_json(Path(prompt_path))["prompts"] if not prompt else [prompt]
    n_satisfying_resp = 0

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

        for _ in range(n_samples):
            for p in prompts:
                satisfied = await make_inference_requests(shards, p, max_tokens)
                n_satisfying_resp += int(satisfied)

    print(f"\nnumber of responses reaching max-tokens: {n_satisfying_resp}")
    print(f"AVG MoE latency: {statistics.mean(STATS['moe_lat'])} ms")
    print(f"AVG Comm latency: {statistics.mean(STATS['comm_lat'])} ms")
    print(f"AVG Misc latency: {statistics.mean(STATS['misc_lat'])} ms")
    print(f"AVG Num Experts Activated Per Node: {statistics.mean(STATS['experts_act'])}")
    print(f"AVG Prompt Eval TP: {statistics.mean(STATS['prompt_eval_tp'])} t/s")
    print(f"AVG Token Gen TP: {statistics.mean(STATS['token_gen_tp'])} t/s")
    pprint.pp(STATS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str)
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Message to be processed by the model",
    )
    parser.add_argument("--prompt-path", type=str)
    parser.add_argument("--n-samples", type=int)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum number of tokens to generate",
    )
    args = parser.parse_args()

    logging.basicConfig()
    asyncio.run(start(args.config_path, args.prompt, args.prompt_path, args.n_samples, args.max_tokens))
