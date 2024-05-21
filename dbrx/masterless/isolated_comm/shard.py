#!/Users/xiangruike/miniconda3/envs/dbrx_poc/bin/python

from contextlib import AsyncExitStack
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

DEFAULT_TEMP = 0.6
DEFAULT_SEED = 7

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

    def design_jobs(self, inds: list[list[int]], dense: bool = False) -> list:
        jobs = []

        if dense:
            activated_experts = set(self.my_experts)
            for _ in range(len(inds)):
                jobs.append([activated_experts, 0])
            return jobs

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
        jobs = self.design_jobs(inds, dense=False)  # CONFIGURABLE

        tic = time.perf_counter_ns()

        expert_outs, arr_map = shard(x, jobs)

        logging.info(f"moe took {(time.perf_counter_ns() - tic) / 1000} micro-sec(s)")
        tic = time.perf_counter_ns()

        shard_outs = self.dispatch_and_combine(expert_outs, arr_map, conn)

        logging.info(f"DnC took {(time.perf_counter_ns() - tic) / 1000} micro-sec(s)")

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
    def __init__(self, args: ModelArgs, experts: mx.array, conn: connection.Connection):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.wte = nn.Embedding(args.vocab_size, args.d_model)
        self.blocks = [DecoderLayer(args) for _ in range(args.n_layers)]
        self.moe_shard = MoeShard(experts)
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


class Generator:

    def __init__(
        self, model_path: str, config_filename: str, conn: connection.Connection
    ) -> None:
        mx.random.seed(DEFAULT_SEED)
        self.model_path = Path(model_path)
        self.model_args = self.get_model_args(config_filename)
        self.conn = conn
        self.model = self.load_model(self.conn)
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

    def load_model(self, conn: connection.Connection) -> DBRX:
        url = self.model_args.ffn_config["shard_url"]
        assigned_experts = self.model_args.ffn_config["shard_map"][url]
        # sample:
        # {0: {"weights": mx.array([0, 1, 2, 3])}}
        experts = {
            e: mx.load(str(self.model_path / f"expert{e}.safetensors"))
            for e in assigned_experts
        }
        mx.eval(experts)

        model = DBRX(self.model_args, experts, conn)
        non_expert_weights = mx.load(str(self.model_path / f"non-expert.safetensors"))
        model.load_weights(list(non_expert_weights.items()))
        mx.eval(model.parameters())
        model.eval()

        return model

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
            prompt = self.conn.recv()
            max_tokens = self.conn.recv()
            res = self.generate(prompt, max_tokens, DEFAULT_TEMP)
            self.conn.send(res)


def shard_main(
    model_path: str, config_filename: str, conn: connection.Connection
) -> None:
    logging.basicConfig(level=logging.INFO)
    generator = Generator(model_path, config_filename, conn)
    logging.info("generator ready")
    generator.start()


class ShardEnvoyServicer(shard_envoy_pb2_grpc.ShardEnvoyServicer):

    def __init__(
        self, model_path: str, config_filename: str, conn: connection.Connection
    ) -> None:
        self.conn = conn
        self.config = self.get_config(model_path, config_filename)

        self.buffers = []
        self.sync_complete_events = []
        for _ in range(self.config["n_layers"]):
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

    async def send(
        self,
        shard: shard_envoy_pb2_grpc.ShardEnvoyStub,
        layer_num: int,
        a_bytes: bytes,
        am_bytes: bytes,
    ):
        await shard.Receive(
            shard_envoy_pb2.ShardOuts(
                url=self.config["url"],
                layer_num=layer_num,
                data=a_bytes,
                arr_map=am_bytes,
            )
        )

    async def all_dispatch(
        self, layer_num: int, oth_shards: list[shard_envoy_pb2_grpc.ShardEnvoyStub]
    ) -> tuple:
        while not self.conn.poll():
            await asyncio.sleep(0.001)
        a_bytes = self.conn.recv_bytes()
        am_bytes = self.conn.recv_bytes()

        tic = time.perf_counter_ns()

        async with asyncio.TaskGroup() as tg:
            for shard in oth_shards:
                tg.create_task(self.send(shard, layer_num, a_bytes, am_bytes))

        logging.info(f"D took {(time.perf_counter_ns() - tic) / 1000} micro-sec(s)")

    def Receive(self, request: shard_envoy_pb2.ShardOuts, context):
        buffer = self.buffers[request.layer_num]
        buffer[request.url] = {"eo_bytes": request.data, "am_bytes": request.arr_map}

        if len(buffer) == len(self.config["oth_urls"]):
            self.sync_complete_events[request.layer_num].set()

        return shard_envoy_pb2.Empty()

    async def Start(self, request: shard_envoy_pb2.UsrIns, context) -> None:
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

            self.conn.send(request.prompt)
            self.conn.send(request.max_tokens)

            for _ in range(request.max_tokens):
                for i in range(self.config["n_layers"]):
                    await self.all_dispatch(i, oth_shards)
                    await self.sync_complete_events[i].wait()

                    for url, d in self.buffers[i].items():
                        self.conn.send(url)
                        self.conn.send_bytes(d["eo_bytes"])
                        self.conn.send_bytes(d["am_bytes"])

                    self.reset_buffer_mechanism(i)

                continue_sig = self.conn.recv()
                if not continue_sig:
                    break

        prompt_time, prompt_t_cnt, gen_time, gen_t_cnt, response = self.conn.recv()

        return shard_envoy_pb2.UsrOuts(
            prompt_time=prompt_time,
            prompt_t_cnt=prompt_t_cnt,
            gen_time=gen_time,
            gen_t_cnt=gen_t_cnt,
            response=response,
        )


async def serve(
    port: int, model_path: str, config_filename: str, conn: connection.Connection
):
    server = grpc.aio.server(
        options=[
            ("grpc.max_send_message_length", -1),
            ("grpc.max_receive_message_length", -1),
        ]
    )
    shard_envoy_pb2_grpc.add_ShardEnvoyServicer_to_server(
        ShardEnvoyServicer(model_path, config_filename, conn), server
    )
    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)
    await server.start()
    logging.info(f"server started, listening on {listen_addr}")

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
