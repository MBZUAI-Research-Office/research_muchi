#!/Users/xiangruike/miniconda3/envs/dbrx_poc/bin/python

from contextlib import AsyncExitStack
import argparse
import asyncio
import logging
import time

import numpy as np

import grpc
import test_expert_pb2
import test_expert_pb2_grpc

NUM_LAYERS = 40
EMBEDDING_LENGTH = 6144
NUM_EXPERTS = 4
TOP_K = 4
DUMMY_NP_DATA = np.random.uniform(-1, 1, EMBEDDING_LENGTH).astype(np.float32)
# EXPERT_CHANNELS = [
#     "169.254.238.2:2000",
#     "169.254.238.4:4000",
#     "169.254.238.5:5000",
#     "169.254.238.6:6000",
# ]
EXPERT_CHANNELS = [
    # "169.254.136.2:2000",
    # "169.254.136.4:4000",
    # "169.254.136.5:5000",
    "169.254.136.6:6000",
]


async def execute_on_expert(stub, block_num, activated_experts, data, outputs):
    out = await stub.Execute(
        test_expert_pb2.Input(
            block_num=block_num,
            activated_experts=activated_experts.tobytes(),
            data=data.tobytes(),
        )
    )
    outputs.append(np.frombuffer(out.data, dtype=np.float32))
    return


async def generate(num_tokens: int):
    async with AsyncExitStack() as es:
        expert_channels = [
            await es.enter_async_context(grpc.aio.insecure_channel(url))
            for url in EXPERT_CHANNELS
        ]
        experts = [
            test_expert_pb2_grpc.ExpertStub(channel) for channel in expert_channels
        ]
        latencies = []

        for i in range(num_tokens):
            token_latency = 0
            for j in range(NUM_LAYERS):
                # chosen_experts = np.random.randint(0, NUM_EXPERTS, size=TOP_K)
                chosen_experts = [0]  # TODO: naming change
                activated_experts = np.array([0, 1, 2, 3])
                outputs = []
                tic = time.perf_counter()

                async with asyncio.TaskGroup() as tg:
                    for k in chosen_experts:
                        tg.create_task(
                            execute_on_expert(
                                experts[k], j, activated_experts, DUMMY_NP_DATA, outputs
                            )
                        )

                token_latency += time.perf_counter() - tic
            latencies.append(token_latency)

        print("=" * 20)
        print(f"NUM LAYERS: {NUM_LAYERS}")
        print(f"EMBEDDING_LENGTH: {EMBEDDING_LENGTH}")
        print(f"TOP K: {TOP_K}")
        print(f"token latencies:\n{latencies}")
        print(f"average: {np.mean(latencies)} sec(s)")
        print("=" * 20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=1)
    args = parser.parse_args()
    logging.basicConfig()
    asyncio.run(generate(args.num_tokens))
