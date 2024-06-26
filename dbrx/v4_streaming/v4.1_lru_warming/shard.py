#!/Users/xiangruike/miniconda3/envs/dbrx_poc/bin/python

from contextlib import AsyncExitStack
from collections import deque, OrderedDict
from dataclasses import dataclass
from multiprocessing import connection
from pathlib import Path
from typing import Any, Optional, Tuple
import argparse
import asyncio
import inspect
import json
import logging
import pickle
import pprint
import multiprocessing
import time

import grpc
import shard_envoy_pb2
import shard_envoy_pb2_grpc

import mlx.core as mx
import mlx.nn as nn

from transformers import AutoTokenizer

from serialization_utils import mx_to_bytes, bytes_to_mx

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


class LruCache(OrderedDict):
    # inspired by:
    # https://docs.python.org/3/library/collections.html#collections.OrderedDict
    # https://stackoverflow.com/questions/21062781/shortest-way-to-get-first-item-of-ordereddict-in-python-3

    def get_lru(self) -> Any:
        k = next(iter(self))
        self.move_to_end(k)
        return k


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
        self.ptr_cache = {}

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

    def __call__(self, x: mx.array, job: set, use_cache: bool) -> tuple[mx.array, dict]:

        def get_weights(e):
            if not use_cache:
                self.ptr_cache[e] = next(self.experts[e]["generator"])

            return self.ptr_cache[e]

        def mlp(x, v1, w1, w2, dst):
            y = (self.act_fn(x @ w1) * (x @ v1)) @ w2
            dst.append(y)

        expert_outs = []
        arr_map = {}

        for e in self.experts:
            v1, w1, w2 = get_weights(e)
            if e in job:
                mlp(x, v1, w1, w2, expert_outs)
                arr_map[e] = len(expert_outs) - 1

        if len(expert_outs) > 0:
            expert_outs = mx.stack(expert_outs, axis=0)
            mx.eval(expert_outs)
        else:
            expert_outs = mx.array(False)

        return expert_outs, arr_map


class DistributedMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_num: int):
        super().__init__()
        self.layer_num = layer_num
        self.d_model = args.d_model

        self.url = args.ffn_config["shard_url"]
        self.expert_map = args.ffn_config["expert_map"]
        self.n_oth_shards = len(args.ffn_config["shard_map"]) - 1
        self.lru_cache = LruCache.fromkeys(args.ffn_config["shard_map"][self.url])

        self.n_experts_in_cluster = args.ffn_config["moe_num_experts"]
        self.num_experts_per_tok = args.ffn_config["moe_top_k"]
        self.router = Router(args.d_model, self.n_experts_in_cluster)

    def design_jobs(self, inds: list[list[int]]) -> list:
        jobs = []
        max_loads = []
        cnt = {e: 0 for e in range(self.n_experts_in_cluster)}

        for activated_experts in inds:
            job = set()
            shard_loads = {}
            for e in activated_experts:
                url = self.expert_map[e]
                if url == self.url:
                    job.add(e)
                cnt[e] += 1
                shard_loads[url] = shard_loads.get(url, 0) + 1

            jobs.append(job)
            max_loads.append(max(shard_loads.values()))

        uses_warmup = any(v == 0 for v in cnt.values())

        for i in range(len(jobs)):
            for e in jobs[i]:
                self.lru_cache.move_to_end(e)

            if not uses_warmup:
                continue

            n_warmups = max_loads[i] - len(jobs[i])
            for _ in range(n_warmups):
                jobs[i].add(self.lru_cache.get_lru())

        return jobs

    def all_dispatch(
        self,
        bi: int,
        expert_outs: mx.array,
        arr_map: dict,
        shard_outs: dict,
        conn: connection.Connection,
    ):
        shard_outs.setdefault(self.url, []).append((expert_outs, arr_map))
        conn.send_bytes(mx_to_bytes(expert_outs))
        conn.send_bytes(pickle.dumps((self.url, self.layer_num, bi, arr_map)))

    def all_combine(
        self,
        batch_size: int,
        shard_outs: dict,
        conn: connection.Connection,
    ):
        for _ in range(batch_size * self.n_oth_shards):
            expert_outs = bytes_to_mx(conn.recv_bytes())
            url, li, bi, arr_map = pickle.loads(conn.recv_bytes())
            shard_outs.setdefault(url, {})[bi] = (expert_outs, arr_map)

    def __call__(
        self,
        x: mx.array,
        shard: MoeShard,
        conn: connection.Connection,
    ) -> mx.array:
        ne = self.num_experts_per_tok
        orig_shape = x.shape  # (sample_size, sequence_length, d_model)
        x = x.reshape(-1, x.shape[-1])  # (sample_size * sequence_length, d_model)

        gates = self.router(x)
        gates = mx.softmax(gates.astype(mx.float32), axis=-1)

        inds = mx.stop_gradient(mx.argpartition(-gates, kth=ne, axis=-1)[:, :ne])
        scores = mx.take_along_axis(gates, inds, axis=-1)
        scores = scores / mx.linalg.norm(scores, ord=1, axis=-1, keepdims=True)
        scores = scores.astype(x.dtype)
        mx.eval(inds, scores)

        inds = inds.tolist()
        jobs = self.design_jobs(inds)
        batch_size = x.shape[0]
        shard_outs = {}

        for bi, xt in enumerate(x):
            expert_outs, arr_map = shard(xt, jobs[bi], bool(bi > 0))
            self.all_dispatch(bi, expert_outs, arr_map, shard_outs, conn)

        self.all_combine(batch_size, shard_outs, conn)
        y = []

        for bi, st, it in zip(range(batch_size), scores, inds):
            yt = []
            for e in it:
                url = self.expert_map[e]
                expert_outs, arr_map = shard_outs[url][bi]
                yt.append(expert_outs[arr_map[e]])

            yt = mx.stack(yt, axis=-1)
            yt = (yt * st).sum(axis=-1)
            y.append(yt)

        y = mx.stack(y, axis=0)
        return y.reshape(orig_shape)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_num: int):
        super().__init__()
        self.ffn = DistributedMoeBlock(args, layer_num)
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
        self.blocks = [DecoderLayer(args, i) for i in range(args.n_layers)]
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

        # h.shape = (sample_size, sequence_length, d_model)
        self.conn.send(h.shape[0] * T)  # let envoy know the batch size
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

        self.x = mx.ones((args.d_model,), dtype=mx.bfloat16)
        self.job = set(args.ffn_config["shard_map"][args.ffn_config["shard_url"]])
        mx.eval(self.x)

    def sync_w_oths(self):
        self.conn.send(True)  # signals that I am ready
        self.conn.recv()  # confirms that everyone else is done

    def __call__(self):
        # warms moe_shard for 1 token
        self.moe_shard.reset_expert_generators()
        for _ in range(self.n_layers):
            self.moe_shard(self.x, self.job, use_cache=False)
            self.sync_w_oths()


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

    # this might not be needed because generator.start() enters an infinite loop
    conn.close()


class DataBuffer:

    def __init__(self, n_layers: int, n_oth_shards: int) -> None:
        self.n_layers = n_layers
        self.n_oth_shards = n_oth_shards

        self.data = []
        for _ in range(self.n_layers + 1):  # last bin for before_warm sync
            self.data.append({})

    def reset(self, li: int):
        self.data[li] = {}

    def put(self, d: Any, li: int, bi: int = 0) -> None:
        self.data[li].setdefault(bi, []).append(d)

    async def wait_til_full(self, li: int, bi: int = 0) -> list[Any]:
        while (bi not in self.data[li]) or (
            len(self.data[li][bi]) != self.n_oth_shards
        ):
            await asyncio.sleep(0)
        return self.data[li][bi]


class ShardEnvoyServicer(shard_envoy_pb2_grpc.ShardEnvoyServicer):

    def __init__(
        self, model_path: str, config_filename: str, conn: connection.Connection
    ) -> None:
        self.conn = conn
        self.config = self.get_config(model_path, config_filename)

        self.gen_queue = deque()
        self.buffer = DataBuffer(self.config["n_layers"], len(self.config["oth_urls"]))

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

    async def sync_w_oths(
        self,
        oth_shards: list,
        li: int = None,
        before_warming: bool = False,
    ) -> None:

        async def signal(shard):
            await shard.SignalReady(shard_envoy_pb2.Identifier(li=li))

        if before_warming:
            li = self.config["n_layers"]  # points to buffer's last bin
        else:
            while not self.conn.poll():
                await asyncio.sleep(0)

            self.conn.recv()  # means warmer is done

        async with asyncio.TaskGroup() as tg:
            for shard in oth_shards:
                tg.create_task(signal(shard))
            tg.create_task(self.buffer.wait_til_full(li))

        self.buffer.reset(li)

    async def all_dispatch(self, oth_shards: list) -> None:

        async def dispatch(shard, data, metadata):
            await shard.Receive(shard_envoy_pb2.ShardOuts(data=data, metadata=metadata))

        while not self.conn.poll():
            await asyncio.sleep(0)

        data = self.conn.recv_bytes()
        metadata = self.conn.recv_bytes()

        async with asyncio.TaskGroup() as tg:
            for shard in oth_shards:
                tg.create_task(dispatch(shard, data, metadata))

    async def all_combine(self, li: int, batch_size: int):
        coros = [self.buffer.wait_til_full(li, bi) for bi in range(batch_size)]
        for coro in asyncio.as_completed(coros):
            arr = await coro
            for d, meta_d in arr:
                self.conn.send_bytes(d)
                self.conn.send_bytes(meta_d)

    def SignalReady(self, request: shard_envoy_pb2.Identifier, context):
        self.buffer.put(True, request.li)
        return shard_envoy_pb2.Empty()

    def Receive(self, request: shard_envoy_pb2.ShardOuts, context):
        url, li, bi, arr_map = pickle.loads(request.metadata)
        self.buffer.put((request.data, request.metadata), li, bi)
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
                    await self.sync_w_oths(oth_shards, before_warming=True)
                    logging.info(f"warming...")
                    self.conn.send(False)  # signal Generator that this is a warming run

                    for li in range(self.config["n_layers"]):
                        await self.sync_w_oths(oth_shards, li=li)
                        self.conn.send(True)  # signals warmer that this layer is done

                    DEFAULT_STARTUP_WARMING_PERIOD -= 1
                    if DEFAULT_STARTUP_WARMING_PERIOD == 0:
                        logging.info(f"completed startup warming")

                    continue

                self.conn.send(True)  # signal Generator that this is a generate run
                gen = self.gen_queue[0]
                self.conn.send(gen["req"].prompt)
                self.conn.send(gen["req"].max_tokens)

                for _ in range(gen["req"].max_tokens):
                    batch_size = self.conn.recv()
                    for li in range(self.config["n_layers"]):
                        for _bi in range(batch_size):
                            await self.all_dispatch(oth_shards)

                        await self.all_combine(li, batch_size)
                        self.buffer.reset(li)

                    continue_sig = self.conn.recv()
                    if not continue_sig:
                        break

                gen["resp"] = self.conn.recv()
                gen["completed"].set()
                self.gen_queue.popleft()


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
