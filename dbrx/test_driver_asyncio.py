import time

import aiohttp
import asyncio

from flask import Flask, request
import numpy as np

NUM_LAYERS = 40
EMBEDDING_LENGTH = 6144
NUM_EXPERTS = 2
TOP_K = 2
EXPERTS = {0: "http://127.0.0.1:3000", 1: "http://127.0.0.1:4000"}
DUMMY_NP_DATA = np.arange(EMBEDDING_LENGTH, dtype=np.uint16).view(np.float16)
app = Flask(__name__)


async def call_expert(session, url, data, experts_out):
    async with session.post(f"{url}/execute", data=data.tobytes()) as response:
        experts_out.append(np.frombuffer(await response.read(), dtype=np.float16))


async def generate_helper():
    latencies = []

    async with aiohttp.ClientSession() as session:
        for i in range(int(request.args.get("num_tokens"))):
            token_latency = 0
            for j in range(NUM_LAYERS):
                # chosen_experts = np.random.randint(0, NUM_EXPERTS, size=TOP_K)
                chosen_experts = [0, 1]
                experts_out = []
                tic = time.perf_counter()

                async with asyncio.TaskGroup() as tg:
                    for k in chosen_experts:
                        tg.create_task(
                            call_expert(session, EXPERTS[k], DUMMY_NP_DATA, experts_out)
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


@app.get("/generate")
def generate():
    asyncio.run(generate_helper())
    return {"status": "success"}
