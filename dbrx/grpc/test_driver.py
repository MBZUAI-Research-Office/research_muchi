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
DUMMY_NP_DATA = np.arange(EMBEDDING_LENGTH, dtype=np.uint16).view(np.float16)
EXPERT_CHANNELS = [
    "localhost:3000",
    "localhost:4000",
    "localhost:5000",
    "localhost:6000",
]


async def execute_on_expert(stub, data, experts_out):
    out = await stub.Execute(test_expert_pb2.Input(data=data.tobytes()))
    experts_out.append(np.frombuffer(out.data, dtype=np.float16))
    return


async def generate(num_tokens: int):
    async with AsyncExitStack() as es:
        expert_channels = [
            await es.enter_async_context(grpc.aio.insecure_channel(url)) for url in EXPERT_CHANNELS
        ]
        experts = {
            i: test_expert_pb2_grpc.ExpertStub(channel)
            for i, channel in enumerate(expert_channels)
        }
        latencies = []

        for i in range(num_tokens):
            token_latency = 0
            for j in range(NUM_LAYERS):
                # chosen_experts = np.random.randint(0, NUM_EXPERTS, size=TOP_K)
                chosen_experts = [0, 1, 2, 3]
                experts_out = []
                tic = time.perf_counter()

                async with asyncio.TaskGroup() as tg:
                    for k in chosen_experts:
                        tg.create_task(
                            execute_on_expert(experts[k], DUMMY_NP_DATA, experts_out)
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
