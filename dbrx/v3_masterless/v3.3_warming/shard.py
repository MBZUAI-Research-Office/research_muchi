#!/Users/xiangruike/miniconda3/envs/dbrx_poc/bin/python

from contextlib import AsyncExitStack
from collections import deque
from dataclasses import dataclass
from multiprocessing import connection
from pathlib import Path
from typing import Optional, Tuple
import argparse
import asyncio
import inspect
import json
import logging
import pickle
import multiprocessing
import time

import grpc
import shard_envoy_pb2
import shard_envoy_pb2_grpc

import mlx.core as mx
import mlx.nn as nn

from transformers import AutoTokenizer

from serialization_utils import mx_to_bytes, bytes_to_mx

import pprint

# from statistics import mean
# LATENCIES = {
#     "moe": [],
#     "comm_0": [],
#     "comm_1": [],
#     "comm_2": [],
#     "comm_3": [],
# }

DEFAULT_TEMP = 0.6
DEFAULT_SEED = 7
DEFAULT_STARTUP_WARMING_PERIOD = 10  # unit: tokens

# coroutines to be invoked when the event loop is shutting down
# copied from:
# https://github.com/grpc/grpc/blob/master/examples/python/helloworld/async_greeter_server_with_graceful_shutdown.py
_cleanup_coroutines = []


@dataclass
class ModelArgs:
    vocab_size: int
    d_model: int
    ffn_config: dict
    attn_config: dict
    n_layers: int
    n_heads: int

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_heads = args.n_heads
        self.d_model = args.d_model
        self.head_dim = args.d_model // args.n_heads
        self.num_key_value_heads = args.attn_config["kv_n_heads"]
        self.clip_qkv = args.attn_config["clip_qkv"]
        self.rope_theta = args.attn_config["rope_theta"]

        self.scale = self.head_dim**-0.5

        self.Wqkv = nn.Linear(
            args.d_model,
            (self.num_key_value_heads * 2 + self.num_heads) * self.head_dim,
            bias=False,
        )
        self.out_proj = nn.Linear(args.d_model, args.d_model, bias=False)
        self.rope = nn.RoPE(
            self.head_dim,
            traditional=False,
            base=self.rope_theta,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:

        qkv = self.Wqkv(x)
        qkv = mx.clip(qkv, a_min=-self.clip_qkv, a_max=self.clip_qkv)
        splits = [self.d_model, self.d_model + self.head_dim * self.num_key_value_heads]
        queries, keys, values = mx.split(qkv, splits, axis=-1)

        B, L, D = x.shape

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )

        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(output), (keys, values)


class NormAttnNorm(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.norm_1 = nn.LayerNorm(args.d_model, bias=False)
        self.norm_2 = nn.LayerNorm(args.d_model, bias=False)
        self.attn = Attention(args)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        h, cache = self.attn(self.norm_1(x), mask=mask, cache=cache)
        x = h + x
        return x, self.norm_2(x), cache


class Router(nn.Module):
    def __init__(self, d_model: int, num_experts: int):
        super().__init__()
        self.layer = nn.Linear(d_model, num_experts, bias=False)

    def __call__(self, x: mx.array):
        return self.layer(x)


class MoeShard:

    def __init__(self, experts: dict) -> None:
        self.experts = experts
        self.act_fn = nn.silu

    def get_expert_generator(self, e: int):
        v1, w1 = None, None
        for i, weight in enumerate(self.experts[e]["weights"]):
            if i % 3 == 0:
                v1 = weight.T
            elif i % 3 == 1:
                w1 = weight.T
            else:
                yield v1, w1, weight

    def reset_expert_generators(self):
        for e in self.experts:
            self.experts[e]["generator"] = self.get_expert_generator(e)

    def __call__(self, inputs: mx.array, jobs: list) -> tuple[mx.array, dict]:
        # sample jobs:
        # [[{14}, 1], [{}, 2]]
        # for each job,
        # job[0] indicates activated experts in this shard for inputs[i]
        # job[1] indicates num additional calculations needed to avoid
        # wire memory driver activity from surfacing

        def mlp(x, v1, w1, w2, dst):
            y = (self.act_fn(x @ w1) * (x @ v1)) @ w2
            dst.append(y)

        expert_outs = []
        arr_map = {}

        for e in self.experts:
            v1, w1, w2 = next(self.experts[e]["generator"])
            for i, x in enumerate(inputs):
                if e in jobs[i][0]:
                    mlp(x, v1, w1, w2, expert_outs)
                    arr_map[f"{i}.{e}"] = len(expert_outs) - 1
                elif jobs[i][1] > 0:
                    mlp(x, v1, w1, w2, expert_outs)
                    jobs[i][1] -= 1

        expert_outs = mx.stack(expert_outs, axis=0)
        mx.eval(expert_outs)

        return expert_outs, arr_map


class DistributedMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.d_model = args.d_model
        self.my_url = args.ffn_config["shard_url"]
        self.my_experts = args.ffn_config["shard_map"][self.my_url]
        self.n_oth_shards = len(args.ffn_config["shard_map"]) - 1
        self.num_experts_per_tok = args.ffn_config["moe_top_k"]
        self.expert_map = args.ffn_config["expert_map"]
        self.router = Router(args.d_model, args.ffn_config["moe_num_experts"])

    def design_jobs(self, inds: list[list[int]]) -> list:
        jobs = []

        for activated_experts in inds:
            job = [set(), 0]
            shard_loads = {}

            for e in activated_experts:
                url = self.expert_map[e]
                if url == self.my_url:
                    job[0].add(e)
                shard_loads[url] = shard_loads.get(url, 0) + 1

            job[1] = max(shard_loads.values()) - len(job[0])
            jobs.append(job)

        return jobs

    def dispatch_and_combine(
        self,
        expert_outs: mx.array,
        arr_map: dict,
        conn: connection.Connection,
    ):
        shard_outs = {}
        shard_outs[self.my_url] = {"expert_outs": expert_outs, "arr_map": arr_map}
        conn.send_bytes(mx_to_bytes(expert_outs))
        conn.send_bytes(pickle.dumps(arr_map))

        for _ in range(self.n_oth_shards):
            oth_url = conn.recv()
            oth_eo = bytes_to_mx(conn.recv_bytes())
            oth_am = pickle.loads(conn.recv_bytes())
            shard_outs[oth_url] = {"expert_outs": oth_eo, "arr_map": oth_am}

        return shard_outs

    def __call__(
        self,
        x: mx.array,
        shard: MoeShard,
        conn: connection.Connection,
    ) -> mx.array:
        ne = self.num_experts_per_tok
        orig_shape = x.shape
        x = x.reshape(-1, x.shape[-1])

        gates = self.router(x)
        gates = mx.softmax(gates.astype(mx.float32), axis=-1)

        inds = mx.stop_gradient(mx.argpartition(-gates, kth=ne, axis=-1)[:, :ne])
        scores = mx.take_along_axis(gates, inds, axis=-1)
        scores = scores / mx.linalg.norm(scores, ord=1, axis=-1, keepdims=True)
        scores = scores.astype(x.dtype)
        mx.eval(inds, scores)

        inds = inds.tolist()
        jobs = self.design_jobs(inds)

        # tic = time.perf_counter_ns()

        expert_outs, arr_map = shard(x, jobs)

        # moe_lat = (time.perf_counter_ns() - tic) / 1000
        # LATENCIES["moe"].append(moe_lat)
        # logging.info(f"moe took {moe_lat} mu_s")
        # tic = time.perf_counter_ns()

        shard_outs = self.dispatch_and_combine(expert_outs, arr_map, conn)

        # comm_3_lat = (time.perf_counter_ns() - tic) / 1000
        # LATENCIES["comm_3"].append(comm_3_lat)
        # logging.info(f"comm_3 took {comm_3_lat} mu_s")

        y = []

        for bi, st, it in zip(range(x.shape[0]), scores, inds):
            yt = []
            for e in it:
                url = self.expert_map[e]
                expert_outs = shard_outs[url]["expert_outs"]
                eoi = shard_outs[url]["arr_map"][f"{bi}.{e}"]
                yt.append(expert_outs[eoi])

            yt = mx.stack(yt, axis=-1)
            yt = (yt * st).sum(axis=-1)
            y.append(yt)

        y = mx.stack(y, axis=0)
        return y.reshape(orig_shape)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.ffn = DistributedMoeBlock(args)
        self.norm_attn_norm = NormAttnNorm(args)

    def __call__(
        self,
        x: mx.array,
        shard: MoeShard,
        conn: connection.Connection,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        r, h, cache = self.norm_attn_norm(x, mask, cache)
        out = self.ffn(h, shard, conn) + r
        return out, cache


class DBRX(nn.Module):
    def __init__(
        self, args: ModelArgs, moe_shard: MoeShard, conn: connection.Connection
    ):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.wte = nn.Embedding(args.vocab_size, args.d_model)
        self.blocks = [DecoderLayer(args) for _ in range(args.n_layers)]
        self.moe_shard = moe_shard
        self.conn = conn
        self.norm_f = nn.LayerNorm(args.d_model, bias=False)
        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        h = self.wte(inputs)

        mask = None
        T = h.shape[1]
        if T > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
            mask = mask.astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.blocks)

        self.moe_shard.reset_expert_generators()
        for e, layer in enumerate(self.blocks):
            h, cache[e] = layer(h, self.moe_shard, self.conn, mask, cache[e])

        return self.lm_head(self.norm_f(h)), cache


class Warmer:
    def __init__(
        self, args: ModelArgs, moe_shard: MoeShard, conn: connection.Connection
    ):
        self.n_layers = args.n_layers
        self.moe_shard = moe_shard
        self.conn = conn

        self.x = mx.ones((1, args.d_model), dtype=mx.bfloat16)
        self.jobs = self.design_jobs(
            args.ffn_config["shard_map"][args.ffn_config["shard_url"]]
        )
        mx.eval(self.x)

    def design_jobs(self, my_experts: list) -> list:
        return [[set(my_experts), 0]]

    def sync_wth_oths(self):
        self.conn.send(True)  # signals that I am ready
        self.conn.recv()  # confirms that everyone else is ready

    def __call__(self):
        # warms moe_shard for 1 token
        self.moe_shard.reset_expert_generators()
        for _ in range(self.n_layers):
            self.moe_shard(self.x, self.jobs)
            self.sync_wth_oths()


class Generator:

    def __init__(
        self, model_path: str, config_filename: str, conn: connection.Connection
    ) -> None:
        mx.random.seed(DEFAULT_SEED)
        self.model_path = Path(model_path)
        self.model_args = self.get_model_args(config_filename)
        self.conn = conn
        self.model, self.warmer = self.load_model_and_warmer()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )

    def get_model_args(self, config_filename: str) -> ModelArgs:
        try:
            with open(self.model_path / config_filename, "r") as f:
                config = json.load(f)
        except FileNotFoundError:
            logging.error(f"{config_filename} not found in {self.model_path}")
            raise

        model_args = ModelArgs.from_dict(config)
        model_args.ffn_config["expert_map"] = {}

        for url, assigned_experts in model_args.ffn_config["shard_map"].items():
            for e in assigned_experts:
                model_args.ffn_config["expert_map"][e] = url

        return model_args

    def load_model_and_warmer(self) -> tuple[DBRX, Warmer]:
        url = self.model_args.ffn_config["shard_url"]
        assigned_experts = self.model_args.ffn_config["shard_map"][url]
        # sample:
        # {0: {"weights": mx.array([0, 1, 2, 3])}}
        experts = {
            e: mx.load(str(self.model_path / f"expert{e}.safetensors"))
            for e in assigned_experts
        }
        mx.eval(experts)
        moe_shard = MoeShard(experts)

        model = DBRX(self.model_args, moe_shard, self.conn)
        non_expert_weights = mx.load(str(self.model_path / f"non-expert.safetensors"))
        model.load_weights(list(non_expert_weights.items()))
        mx.eval(model.parameters())
        model.eval()

        return model, Warmer(self.model_args, moe_shard, self.conn)

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temp: float,
    ):

        def sample(logits: mx.array) -> Tuple[mx.array, float]:
            # softmax_logits = mx.softmax(logits)

            if temp == 0:
                token = mx.argmax(logits, axis=-1)
            else:
                token = mx.random.categorical(logits * (1 / temp))

            # prob = softmax_logits[0, token]
            # return token, prob
            return token

        prompt_tokens = mx.array(self.tokenizer.encode(prompt))
        y = prompt_tokens
        cache = None
        tokens = []
        token_strings = []
        REPLACEMENT_CHAR = "\ufffd"

        tic = time.perf_counter()

        for n in range(max_tokens):
            logits, cache = self.model(y[None], cache=cache)
            logits = logits[:, -1, :]
            y = sample(logits)

            token = y.item()  # get word ID
            if n == 0:
                prompt_time = time.perf_counter() - tic
                tic = time.perf_counter()
            if token == self.tokenizer.eos_token_id:
                self.conn.send(False)
                break
            tokens.append(token)

            s = self.tokenizer.decode(tokens)  # str
            # Reset token cache at line break
            if s[-1] == "\n":
                tokens = []
                token_strings.append(s)

            self.conn.send(True)

        token_count = n + 1
        token_strings.append(
            self.tokenizer.decode(tokens).replace(REPLACEMENT_CHAR, "")
        )
        gen_time = time.perf_counter() - tic

        return (
            prompt_time,
            prompt_tokens.size,
            gen_time,
            token_count,
            "".join(token_strings),
        )

    def start(self) -> None:
        while True:
            is_gen_run = self.conn.recv()

            if not is_gen_run:
                self.warmer()
                continue

            prompt = self.conn.recv()
            max_tokens = self.conn.recv()
            res = self.generate(prompt, max_tokens, DEFAULT_TEMP)
            # logging.info(f"avg moe latency: {mean(LATENCIES['moe'][40:])} mu_s")
            # logging.info(f"avg comm_3 latency: {mean(LATENCIES['comm_3'][40:])} mu_s")
            pprint.pp(res)
            self.conn.send(res)


def shard_main(
    model_path: str, config_filename: str, conn: connection.Connection
) -> None:
    logging.basicConfig(level=logging.INFO)
    generator = Generator(model_path, config_filename, conn)
    conn.send(True)  # signals envoy that it is ready
    logging.info("generator ready")
    generator.start()


class ShardEnvoyServicer(shard_envoy_pb2_grpc.ShardEnvoyServicer):

    def __init__(
        self, model_path: str, config_filename: str, conn: connection.Connection
    ) -> None:
        self.conn = conn
        self.config = self.get_config(model_path, config_filename)

        self.gen_queue = deque()
        self.buffers = []
        self.sync_complete_events = []
        for _ in range(self.config["n_layers"] + 1):  # one extra for before_warm signal
            self.buffers.append({})
            self.sync_complete_events.append(asyncio.Event())

    def reset_buffer_mechanism(self, i: int):
        self.buffers[i] = {}
        self.sync_complete_events[i].clear()

    def get_config(self, model_path: str, config_filename: str) -> dict:
        try:
            with open(Path(model_path) / config_filename, "r") as f:
                tmp = json.load(f)
        except FileNotFoundError:
            logging.error(f"{config_filename} not found in {model_path}")
            raise

        config = {
            "n_layers": tmp["n_layers"],
            "url": tmp["ffn_config"]["shard_url"],
            "oth_urls": [],
        }

        for url in tmp["ffn_config"]["shard_map"]:
            if url == config["url"]:
                continue
            config["oth_urls"].append(url)

        return config

    async def broadcast_im_ready(
        self,
        layer_num: int,
        oth_shards: list[shard_envoy_pb2_grpc.ShardEnvoyStub],
        before_warming: bool,
    ):

        async def signal(shard):
            await shard.WarmingSync(
                shard_envoy_pb2.Identifier(url=self.config["url"], layer_num=layer_num)
            )

        if not before_warming:
            while not self.conn.poll():
                await asyncio.sleep(0)

            self.conn.recv()  # means warmer is done

        async with asyncio.TaskGroup() as tg:
            for shard in oth_shards:
                tg.create_task(signal(shard))

    async def all_dispatch(
        self, layer_num: int, oth_shards: list[shard_envoy_pb2_grpc.ShardEnvoyStub]
    ):

        async def send(shard, a_bytes, am_bytes):
            await shard.Receive(
                shard_envoy_pb2.ShardOuts(
                    url=self.config["url"],
                    layer_num=layer_num,
                    data=a_bytes,
                    arr_map=am_bytes,
                )
            )

        # tic = time.perf_counter_ns()

        while not self.conn.poll():
            await asyncio.sleep(0)

        # comm_0_lat = (time.perf_counter_ns() - tic) / 1000
        # LATENCIES["comm_0"].append(comm_0_lat)
        # logging.info(f"comm_0 took {comm_0_lat} mu_s")

        a_bytes = self.conn.recv_bytes()
        am_bytes = self.conn.recv_bytes()

        # tic = time.perf_counter_ns()

        async with asyncio.TaskGroup() as tg:
            for shard in oth_shards:
                tg.create_task(send(shard, a_bytes, am_bytes))

        # comm_1_lat = (time.perf_counter_ns() - tic) / 1000
        # LATENCIES["comm_1"].append(comm_1_lat)
        # logging.info(f"comm_1 took {comm_1_lat} mu_s")

    def signal_if_sync_completed(self, i: int):
        if len(self.buffers[i]) == len(self.config["oth_urls"]):
            self.sync_complete_events[i].set()

    def WarmingSync(self, request: shard_envoy_pb2.Identifier, context):
        self.buffers[request.layer_num][request.url] = True
        self.signal_if_sync_completed(request.layer_num)
        return shard_envoy_pb2.Empty()

    def Receive(self, request: shard_envoy_pb2.ShardOuts, context):
        self.buffers[request.layer_num][request.url] = {
            "eo_bytes": request.data,
            "am_bytes": request.arr_map,
        }
        self.signal_if_sync_completed(request.layer_num)
        return shard_envoy_pb2.Empty()

    async def Generate(self, request: shard_envoy_pb2.UsrIns, context):
        logging.info(f"received generation request")
        job = {"req": request, "completed": asyncio.Event()}
        self.gen_queue.append(job)
        await job["completed"].wait()
        return shard_envoy_pb2.UsrOuts(
            prompt_time=job["resp"][0],
            prompt_t_cnt=job["resp"][1],
            gen_time=job["resp"][2],
            gen_t_cnt=job["resp"][3],
            response=job["resp"][4],
        )

    async def start(self) -> None:

        global DEFAULT_STARTUP_WARMING_PERIOD

        async def stay_warm(i, oth_shards, before_warming=False):
            await self.broadcast_im_ready(i, oth_shards, before_warming)
            await self.sync_complete_events[i].wait()
            self.reset_buffer_mechanism(i)

        async with AsyncExitStack() as es:
            oth_shards = []
            for url in self.config["oth_urls"]:
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
                oth_shards.append(shard)

            while True:
                if len(self.gen_queue) == 0 or DEFAULT_STARTUP_WARMING_PERIOD > 0:

                    await stay_warm(self.config["n_layers"], oth_shards, before_warming=True)
                    logging.info(f"warming...")
                    self.conn.send(False)  # signal Generator that this is a warming run

                    for i in range(self.config["n_layers"]):
                        await stay_warm(i, oth_shards)

                    DEFAULT_STARTUP_WARMING_PERIOD -= 1
                    if DEFAULT_STARTUP_WARMING_PERIOD == 0:
                        logging.info(f"completed startup warming")

                    continue

                self.conn.send(True)  # signal Generator that this is a generate run
                gen = self.gen_queue[0]
                self.conn.send(gen["req"].prompt)
                self.conn.send(gen["req"].max_tokens)

                for _ in range(gen["req"].max_tokens):
                    for i in range(self.config["n_layers"]):
                        # tic = time.perf_counter_ns()

                        await self.all_dispatch(i, oth_shards)
                        await self.sync_complete_events[i].wait()

                        # comm_2_lat = (time.perf_counter_ns() - tic) / 1000
                        # LATENCIES["comm_2"].append(comm_2_lat)
                        # logging.info(f"comm_2 took {comm_2_lat} mu_s")

                        for url, d in self.buffers[i].items():
                            self.conn.send(url)
                            self.conn.send_bytes(d["eo_bytes"])
                            self.conn.send_bytes(d["am_bytes"])

                        self.reset_buffer_mechanism(i)

                    continue_sig = self.conn.recv()
                    if not continue_sig:
                        break

                gen["resp"] = self.conn.recv()
                gen["completed"].set()
                self.gen_queue.popleft()

                # logging.info(f"avg comm_0 latency: {mean(LATENCIES['comm_0'][40:])} mu_s")
                # logging.info(f"avg comm_1 latency: {mean(LATENCIES['comm_1'][40:])} mu_s")
                # logging.info(f"avg comm_2 latency: {mean(LATENCIES['comm_2'][40:])} mu_s")


async def serve(
    port: int, model_path: str, config_filename: str, conn: connection.Connection
):
    server = grpc.aio.server(
        options=[
            ("grpc.max_send_message_length", -1),
            ("grpc.max_receive_message_length", -1),
        ]
    )
    servicer = ShardEnvoyServicer(model_path, config_filename, conn)
    shard_envoy_pb2_grpc.add_ShardEnvoyServicer_to_server(servicer, server)
    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)
    await server.start()
    logging.info(f"server started, listening on {listen_addr}")

    conn.recv()  # wait for Generator to finish initializing before starting to warm up
    await servicer.start()

    # copied from:
    # https://github.com/grpc/grpc/blob/master/examples/python/helloworld/async_greeter_server_with_graceful_shutdown.py
    async def server_graceful_shutdown():
        logging.info("Starting graceful shutdown...")
        # Shuts down the server with 3 seconds of grace period. During the
        # grace period, the server won't accept new connections and allow
        # existing RPCs to continue within the grace period.
        await server.stop(3)

    _cleanup_coroutines.append(server_graceful_shutdown())
    await server.wait_for_termination()


def envoy_main(
    port: int, model_path: str, config_filename: str, conn: connection.Connection
):
    logging.basicConfig(level=logging.INFO)
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(serve(port, model_path, config_filename, conn))
    finally:
        loop.run_until_complete(*_cleanup_coroutines)
        loop.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int)
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--config-filename", type=str)
    args = parser.parse_args()

    envoy_conn, shard_conn = multiprocessing.Pipe()

    envoy_p = multiprocessing.Process(
        target=envoy_main,
        args=(args.port, args.model_path, args.config_filename, envoy_conn),
    )
    shard_p = multiprocessing.Process(
        target=shard_main, args=(args.model_path, args.config_filename, shard_conn)
    )

    envoy_p.start()
    shard_p.start()

    envoy_p.join()
    shard_p.join()
