from multiprocessing import Pool
import time

from flask import Flask, request
import numpy as np
import requests

NUM_LAYERS = 40
EMBEDDING_LENGTH = 6144
NUM_EXPERTS = 2
TOP_K = 2
EXPERTS = {
    0: "169.254.238.2:2000",
    1: "169.254.238.4:4000",
    2: "169.254.238.5:5000",
    3: "169.254.238.6:6000",
}
DUMMY_NP_DATA = np.arange(EMBEDDING_LENGTH, dtype=np.uint16).view(np.float16)
app = Flask(__name__)


def call_expert(args):
    resp = requests.post(f"{args['url']}/execute", data=args["data"].tobytes())
    return np.frombuffer(resp.content, dtype=np.float16)


@app.get("/generate")
def generate():
    latencies = []

    with Pool(2) as pool:
        for i in range(int(request.args.get("num_tokens"))):
            token_latency = 0
            for j in range(NUM_LAYERS):
                # chosen_experts = np.random.randint(0, NUM_EXPERTS, size=TOP_K)
                chosen_experts = [0, 1, 2, 3]
                tic = time.perf_counter()

                pool.map(
                    call_expert,
                    [
                        {"url": EXPERTS[k], "data": DUMMY_NP_DATA}
                        for k in chosen_experts
                    ],
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

    return {"status": "success"}
