#!/Users/xiangruike/miniconda3/envs/dbrx_enhanced/bin/python

from contextlib import AsyncExitStack
from collections import deque, OrderedDict
from dataclasses import dataclass
from multiprocessing import connection
from pathlib import Path
from typing import Any, Optional, Tuple
import concurrent.futures
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

DEFAULT_TEMP = 0.6
DEFAULT_SEED = 7
DEFAULT_STARTUP_WARMING_PERIOD = 10  # unit: tokens

# coroutines to be invoked when the event loop is shutting down
# copied from:
# https://github.com/grpc/grpc/blob/master/examples/python/helloworld/async_greeter_server_with_graceful_shutdown.py
_cleanup_coroutines = []

import statistics

LOGS = {
    "moe_lat": [],
    "comm_lat": [],
    "experts_act": [],
}


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


class RawWeights:

    def __init__(
        self,
        n_layers: int,
        wqkv: mx.array,
        out_proj: mx.array,
        experts: dict,
        lm_head: mx.array,
    ) -> None:
        ptrs = {i: {} for i in range(n_layers)}
        for i, mat in enumerate(wqkv["weights"]):
            ptrs[i]["wqkv"] = mat
        for i, mat in enumerate(out_proj["weights"]):
            ptrs[i]["out_proj"] = mat
        for e, d in experts.items():
            for j, mat in enumerate(d["weights"]):
                i = j // 3
                if e not in ptrs[i]:
                    ptrs[i][e] = {}
                if j % 3 == 0:
                    ptrs[i][e]["v1"] = mat
                elif j % 3 == 1:
                    ptrs[i][e]["w1"] = mat
                else:
                    ptrs[i][e]["w2"] = mat
        ptrs["lm_head"] = lm_head

        standby_extras = []
        light_extras = []
        for vec in ptrs[0]["wqkv"]:
            standby_extras.append(vec)
            light_extras.append(vec)
            break
        for vec in ptrs[0]["out_proj"]:
            standby_extras.append(vec)
            light_extras.append(vec)
            break
        for e in experts:
            for vec in ptrs[0][e]["v1"]:
                standby_extras.append(vec)
                break
        for vec in ptrs["lm_head"]:
            standby_extras.append(vec)
            light_extras.append(vec)
            break

        self.ptrs = ptrs
        self.standby_extras = standby_extras
        self.light_extras = light_extras

    def __call__(self, k):
        return self.ptrs[k]


class LruCache(OrderedDict):
    # inspired by:
    # https://docs.python.org/3/library/collections.html#collections.OrderedDict
    # https://stackoverflow.com/questions/21062781/shortest-way-to-get-first-item-of-ordereddict-in-python-3

    def get_lru(self) -> Any:
        k = next(iter(self))
        self.move_to_end(k)
        return k


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_num: int):
        super().__init__()
        self.layer_num = layer_num
        self.num_heads = args.n_heads
        self.d_model = args.d_model
        self.head_dim = args.d_model // args.n_heads
        self.num_key_value_heads = args.attn_config["kv_n_heads"]
        self.clip_qkv = args.attn_config["clip_qkv"]

        self.scale = self.head_dim**-0.5
        self.rope = nn.RoPE(
            self.head_dim,
            traditional=False,
            base=args.attn_config["rope_theta"],
        )

    def __call__(
        self,
        x: mx.array,
        raw_weights: RawWeights,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        ws = raw_weights(self.layer_num)
        qkv = x @ ws["wqkv"].T
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
        return output @ ws["out_proj"].T, (keys, values)


class NormAttnNorm(nn.Module):
    def __init__(self, args: ModelArgs, layer_num: int):
        super().__init__()
        self.norm_1 = nn.LayerNorm(args.d_model, bias=False)
        self.norm_2 = nn.LayerNorm(args.d_model, bias=False)
        self.attn = Attention(args, layer_num)

    def __call__(
        self,
        x: mx.array,
        raw_weights: RawWeights,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        h, cache = self.attn(self.norm_1(x), raw_weights, mask=mask, cache=cache)
        x = h + x
        return x, self.norm_2(x), cache


class Router(nn.Module):
    def __init__(self, d_model: int, num_experts: int):
        super().__init__()
        self.layer = nn.Linear(d_model, num_experts, bias=False)

    def __call__(self, x: mx.array):
        return self.layer(x)


class DistributedMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_num: int):
        super().__init__()
        self.act_fn = nn.silu
        self.layer_num = layer_num
        self.d_model = args.d_model

        self.url = args.ffn_config["shard_url"]
        self.experts = args.ffn_config["assigned_experts"]
        self.e_to_g = args.ffn_config["e_to_g"]
        self.dlb_groups = args.ffn_config["dlb_groups"]
        self.n_oth_shards = sum(len(d["members"]) for d in self.dlb_groups.values()) - 1
        self.lru_cache = LruCache.fromkeys(self.experts)

        self.n_experts_in_cluster = args.ffn_config["moe_num_experts"]
        self.num_experts_per_tok = args.ffn_config["moe_top_k"]
        self.router = Router(args.d_model, self.n_experts_in_cluster)

    def allocate_jobs(self, scores: mx.array, inds: list[list[int]]) -> list[dict]:
        jobs = []
        max_loads = []

        for st, it in zip(scores, inds):
            by_dlb_group = {}
            by_shard = {}
            e_to_s = {}

            for s, e in zip(st, it):
                by_dlb_group.setdefault(self.e_to_g[e], []).append(e)
                e_to_s[e] = s

            for g, es in by_dlb_group.items():
                members = self.dlb_groups[g]["members"]
                for i, e in enumerate(es):
                    url = members[i % len(members)]
                    by_shard.setdefault(url, {})[e] = e_to_s[e]

            jobs.append(by_shard.get(self.url, {}))
            max_loads.append(max(len(v) for v in by_shard.values()))

        for i in range(len(jobs)):
            for e in jobs[i]:
                self.lru_cache.move_to_end(e)

            n_warmups = max_loads[i] - len(jobs[i])
            for _ in range(n_warmups):
                jobs[i][self.lru_cache.get_lru()] = 0

            LOGS["experts_act"].append(len(jobs[i]))

        return jobs

    def moe_shard(self, x: mx.array, job: dict, ws: dict) -> mx.array:
        expert_outs = []
        for e in self.experts:
            if e not in job:
                continue
            y = (self.act_fn(x @ ws[e]["w1"].T) * (x @ ws[e]["v1"].T)) @ ws[e]["w2"]
            expert_outs.append(y * job[e])  # multiply by score

        return mx.stack(expert_outs, axis=-1).sum(axis=-1)

    def call_shard_n_all_dispatch(
        self,
        x: mx.array,
        jobs: list[set],
        raw_weights: RawWeights,
        send_conn: connection.Connection,
    ) -> dict:
        tic = time.perf_counter_ns()

        ws = raw_weights(self.layer_num)
        y = []
        for bi, xt in enumerate(x):
            expert_outs = self.moe_shard(xt, jobs[bi], ws, extras)
            extras = mx.sum(mx.stack(raw_weights.light_extras, axis=0), axis=0)
            mx.eval(expert_outs, extras)
            y.append(expert_outs)
            send_conn.send_bytes(mx_to_bytes(expert_outs))
            send_conn.send_bytes(pickle.dumps((self.layer_num, bi)))

        return y, time.perf_counter_ns() - tic

    def all_combine(
        self,
        batch_size: int,
        resv_conn: connection.Connection,
    ):
        y = []
        for _i in range(batch_size):
            yt = []
            for _j in range(self.n_oth_shards):
                expert_outs = bytes_to_mx(resv_conn.recv_bytes())
                yt.append(expert_outs)

            yt = mx.stack(yt, axis=-1).sum(axis=-1)
            y.append(yt)

        return y

    def __call__(
        self,
        x: mx.array,
        raw_weights: RawWeights,
        resv_conn: connection.Connection,
        send_conn: connection.Connection,
        executor: concurrent.futures.ThreadPoolExecutor,
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
        jobs = self.allocate_jobs(scores, inds)

        tic = time.perf_counter_ns()

        compute_fut = executor.submit(
            self.call_shard_n_all_dispatch, x, jobs, raw_weights, send_conn
        )
        comm_fut = executor.submit(self.all_combine, x.shape[0], resv_conn)

        moe_lat = compute_fut.result()[1]
        LOGS["moe_lat"].append(moe_lat)
        LOGS["comm_lat"].append(time.perf_counter_ns() - tic - moe_lat)

        y = [a + b for a, b in zip(compute_fut.result()[0], comm_fut.result())]
        return mx.stack(y, axis=0).reshape(orig_shape)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_num: int):
        super().__init__()
        self.ffn = DistributedMoeBlock(args, layer_num)
        self.norm_attn_norm = NormAttnNorm(args, layer_num)

    def __call__(
        self,
        x: mx.array,
        raw_weights: RawWeights,
        resv_conn: connection.Connection,
        send_conn: connection.Connection,
        executor: concurrent.futures.ThreadPoolExecutor,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        r, h, cache = self.norm_attn_norm(x, raw_weights, mask, cache)
        out = self.ffn(h, raw_weights, resv_conn, send_conn, executor) + r
        return out, cache


class DBRX(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        raw_weights: RawWeights,
        resv_conn: connection.Connection,
        send_conn: connection.Connection,
    ):
        super().__init__()
        self.raw_weights = raw_weights
        self.wte = nn.Embedding(args.vocab_size, args.d_model)
        self.blocks = [DecoderLayer(args, i) for i in range(args.n_layers)]
        self.resv_conn = resv_conn
        self.send_conn = send_conn
        self.norm_f = nn.LayerNorm(args.d_model, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        executor: concurrent.futures.ThreadPoolExecutor,
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
        self.send_conn.send(h.shape[0] * T)  # let envoy know the batch size

        for e, layer in enumerate(self.blocks):
            h, cache[e] = layer(
                h,
                self.raw_weights,
                self.resv_conn,
                self.send_conn,
                executor,
                mask,
                cache[e],
            )

        return self.norm_f(h) @ self.raw_weights("lm_head").T, cache


class Warmer:
    def __init__(
        self,
        args: ModelArgs,
        raw_weights: RawWeights,
        resv_conn: connection.Connection,
        send_conn: connection.Connection,
    ):
        self.n_layers = args.n_layers
        self.raw_weights = raw_weights
        self.resv_conn = resv_conn
        self.send_conn = send_conn

    def warm(self) -> None:
        y = mx.sum(mx.stack(self.raw_weights.standby_extras, axis=0), axis=0)
        mx.eval(y)

    def sync_w_oths(self):
        self.send_conn.send(True)  # signals that I am ready
        self.resv_conn.recv()  # confirms that everyone else is done

    def __call__(self):
        for _ in range(self.n_layers):
            self.warm()
            self.sync_w_oths()


class Generator:

    def __init__(
        self,
        model_path: str,
        config_filename: str,
        resv_conn: connection.Connection,
        send_conn: connection.Connection,
    ) -> None:
        mx.random.seed(DEFAULT_SEED)
        self.model_path = Path(model_path)
        self.model_args = self.get_model_args(config_filename)
        self.resv_conn = resv_conn
        self.send_conn = send_conn
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
        model_args.ffn_config["e_to_g"] = {}

        for g, d in model_args.ffn_config["dlb_groups"].items():
            for e in d["experts"]:
                model_args.ffn_config["e_to_g"][e] = g

            if model_args.ffn_config["shard_url"] in d["members"]:
                model_args.ffn_config["assigned_experts"] = d["experts"]

        return model_args

    def load_model_and_warmer(self) -> tuple[DBRX, Warmer]:
        wqkv = mx.load(str(self.model_path / f"wqkv.safetensors"))
        out_proj = mx.load(str(self.model_path / f"out_proj.safetensors"))
        oth_non_es = mx.load(str(self.model_path / f"non-expert.safetensors"))
        # sample:
        # {0: {"weights": mx.array([0, 1, 2, 3])}}
        experts = {
            e: mx.load(str(self.model_path / f"expert{e}.safetensors"))
            for e in self.model_args.ffn_config["assigned_experts"]
        }
        mx.eval(wqkv, out_proj, oth_non_es, experts)

        raw_weights = RawWeights(
            self.model_args.n_layers,
            wqkv,
            out_proj,
            experts,
            oth_non_es.pop("lm_head.weight"),
        )
        model = DBRX(self.model_args, raw_weights, self.resv_conn, self.send_conn)
        model.load_weights(list(oth_non_es.items()))
        # mx.eval(model.parameters())
        model.eval()

        return model, Warmer(
            self.model_args, raw_weights, self.resv_conn, self.send_conn
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temp: float,
        executor: concurrent.futures.ThreadPoolExecutor,
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
            logits, cache = self.model(y[None], executor, cache=cache)
            logits = logits[:, -1, :]
            y = sample(logits)

            token = y.item()  # get word ID
            if n == 0:
                prompt_time = time.perf_counter() - tic
                tic = time.perf_counter()
            if token == self.tokenizer.eos_token_id:
                self.send_conn.send(False)
                break
            tokens.append(token)

            s = self.tokenizer.decode(tokens)  # str
            # Reset token cache at line break
            if s[-1] == "\n":
                tokens = []
                token_strings.append(s)

            self.send_conn.send(True)

        token_count = n + 1
        token_strings.append(
            self.tokenizer.decode(tokens).replace(REPLACEMENT_CHAR, "")
        )
        gen_time = time.perf_counter() - tic

        return [
            prompt_time,
            prompt_tokens.size,
            gen_time,
            token_count,
            "".join(token_strings),
        ]

    def start(self) -> None:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            while True:
                is_gen_run = self.resv_conn.recv()

                if not is_gen_run:
                    self.warmer()
                    continue

                prompt = self.resv_conn.recv()
                max_tokens = self.resv_conn.recv()
                res = self.generate(prompt, max_tokens, DEFAULT_TEMP, executor)

                # pprint.pp(res)
                for k in ["moe_lat", "comm_lat", "experts_act"]:
                    if len(LOGS[k]) <= 40:
                        # no token generated
                        res.append(0)
                        continue
                    avg = statistics.mean(LOGS[k][40:])
                    if k != "experts_act":
                        avg /= 1000**2
                    res.append(avg)
                    LOGS[k] = []

                self.send_conn.send(res)


def shard_main(
    model_path: str,
    config_filename: str,
    resv_conn: connection.Connection,
    send_conn: connection.Connection,
) -> None:
    logging.basicConfig(level=logging.INFO)
    generator = Generator(model_path, config_filename, resv_conn, send_conn)
    send_conn.send(True)  # signals envoy that it is ready
    logging.info("generator ready")
    generator.start()

    # this might not be needed because generator.start() enters an infinite loop
    resv_conn.close()
    send_conn.close()


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
        self,
        model_path: str,
        config_filename: str,
        resv_conn: connection.Connection,
        send_conn: connection.Connection,
    ) -> None:
        self.resv_conn = resv_conn
        self.send_conn = send_conn
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

        for d in tmp["ffn_config"]["dlb_groups"].values():
            for url in d["members"]:
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
            while not self.resv_conn.poll():
                await asyncio.sleep(0)

            self.resv_conn.recv()  # means warmer is done

        async with asyncio.TaskGroup() as tg:
            for shard in oth_shards:
                tg.create_task(signal(shard))
            tg.create_task(self.buffer.wait_til_full(li))

        self.buffer.reset(li)

    async def all_dispatch_n_combine(self, li: int, bi: int, oth_shards: list) -> None:

        async def dispatch(shard, data, metadata):
            await shard.Receive(shard_envoy_pb2.ShardOuts(data=data, metadata=metadata))

        while not self.resv_conn.poll():
            await asyncio.sleep(0)

        data = self.resv_conn.recv_bytes()
        metadata = self.resv_conn.recv_bytes()

        async with asyncio.TaskGroup() as tg:
            for shard in oth_shards:
                tg.create_task(dispatch(shard, data, metadata))
            wait_task = tg.create_task(self.buffer.wait_til_full(li, bi))

        for d in wait_task.result():
            self.send_conn.send_bytes(d)

    def SignalReady(self, request: shard_envoy_pb2.Identifier, context):
        self.buffer.put(True, request.li)
        return shard_envoy_pb2.Empty()

    def Receive(self, request: shard_envoy_pb2.ShardOuts, context):
        li, bi = pickle.loads(request.metadata)
        self.buffer.put(request.data, li, bi)
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
            avg_moe_lat=job["resp"][5],
            avg_comm_lat=job["resp"][6],
            avg_experts_act=job["resp"][7],
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
                    self.send_conn.send(
                        False
                    )  # signal Generator that this is a warming run

                    for li in range(self.config["n_layers"]):
                        await self.sync_w_oths(oth_shards, li=li)
                        self.send_conn.send(
                            True
                        )  # signals warmer that this layer is done

                    DEFAULT_STARTUP_WARMING_PERIOD -= 1
                    if DEFAULT_STARTUP_WARMING_PERIOD == 0:
                        logging.info(f"completed startup warming")

                    continue

                self.send_conn.send(
                    True
                )  # signal Generator that this is a generate run
                gen = self.gen_queue[0]
                self.send_conn.send(gen["req"].prompt)
                self.send_conn.send(gen["req"].max_tokens)

                for _ in range(gen["req"].max_tokens):
                    batch_size = self.resv_conn.recv()
                    for li in range(self.config["n_layers"]):
                        for bi in range(batch_size):
                            await self.all_dispatch_n_combine(li, bi, oth_shards)

                        self.buffer.reset(li)

                    continue_sig = self.resv_conn.recv()
                    if not continue_sig:
                        break

                gen["resp"] = self.resv_conn.recv()
                gen["completed"].set()
                self.gen_queue.popleft()


async def serve(
    port: int,
    model_path: str,
    config_filename: str,
    resv_conn: connection.Connection,
    send_conn: connection.Connection,
):
    server = grpc.aio.server(
        options=[
            ("grpc.max_send_message_length", -1),
            ("grpc.max_receive_message_length", -1),
        ]
    )
    servicer = ShardEnvoyServicer(model_path, config_filename, resv_conn, send_conn)
    shard_envoy_pb2_grpc.add_ShardEnvoyServicer_to_server(servicer, server)
    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)
    await server.start()
    logging.info(f"server started, listening on {listen_addr}")

    resv_conn.recv()  # wait for Generator to finish initializing before starting to warm up
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
    port: int,
    model_path: str,
    config_filename: str,
    resv_conn: connection.Connection,
    send_conn: connection.Connection,
):
    logging.basicConfig(level=logging.INFO)
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(
            serve(port, model_path, config_filename, resv_conn, send_conn)
        )
    finally:
        loop.run_until_complete(*_cleanup_coroutines)
        loop.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int)
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--config-filename", type=str)
    args = parser.parse_args()

    envoy_resv, envoy_send = multiprocessing.Pipe(duplex=False)
    shard_resv, shard_send = multiprocessing.Pipe(duplex=False)

    envoy_p = multiprocessing.Process(
        target=envoy_main,
        args=(args.port, args.model_path, args.config_filename, shard_resv, envoy_send),
    )
    shard_p = multiprocessing.Process(
        target=shard_main,
        args=(args.model_path, args.config_filename, envoy_resv, shard_send),
    )

    envoy_p.start()
    shard_p.start()

    envoy_p.join()
    shard_p.join()
