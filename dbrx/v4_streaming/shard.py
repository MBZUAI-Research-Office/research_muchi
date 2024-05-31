#!/Users/xiangruike/miniconda3/envs/dbrx_poc/bin/python

from collections import deque
from contextlib import ExitStack
from concurrent import futures
from dataclasses import dataclass
from multiprocessing import connection
from pathlib import Path
from typing import Any, Optional, Tuple
import argparse
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

    def __call__(
        self, x: mx.array, job: list, use_cache: bool
    ) -> tuple[mx.array, dict]:
        # sample job:
        # [{14}, 1]
        # job[0] indicates activated experts in this shard for x
        # job[1] indicates num additional calculations needed to avoid
        # wire memory driver activity from surfacing

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
            if e in job[0]:
                mlp(x, v1, w1, w2, expert_outs)
                arr_map[e] = len(expert_outs) - 1
            elif job[1] > 0:
                mlp(x, v1, w1, w2, expert_outs)
                job[1] -= 1

        expert_outs = mx.stack(expert_outs, axis=0)
        mx.eval(expert_outs)

        return expert_outs, arr_map


class DistributedMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_num: int):
        super().__init__()
        self.layer_num = layer_num
        self.d_model = args.d_model
        self.url = args.ffn_config["shard_url"]
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
                if url == self.url:
                    job[0].add(e)
                shard_loads[url] = shard_loads.get(url, 0) + 1

            job[1] = max(shard_loads.values()) - len(job[0])
            jobs.append(job)

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
            metadata = pickle.loads(conn.recv_bytes())
            shard_outs.setdefault(metadata[0], []).append((expert_outs, metadata[3]))

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
                expert_outs, arr_map = shard_outs[self.expert_map[e]][bi]
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
        self.job = [set(args.ffn_config["shard_map"][args.ffn_config["shard_url"]]), 0]
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


class Buffer:

    def __init__(self, n_layers: int, n_oth_shards: int) -> None:
        self.n_layers = n_layers
        self.n_oth_shards = n_oth_shards
        # last bin reserved for everyone is ready signal
        self.data = [{} for _ in range(self.n_layers + 1)]

    def reset(self, layer_num: int) -> None:
        self.data[layer_num] = {}

    def put(self, d: Any, layer_num: int, bi: int = 0) -> None:
        self.data[layer_num].setdefault(bi, []).append(d)

    def wait_until_full(self, layer_num: int, bi: int = 0) -> list[Any]:
        while (
            bi not in self.data[layer_num]
            or len(self.data[layer_num][bi]) < self.n_oth_shards
        ):
            pass
        return self.data[layer_num][bi]


class ShardEnvoyServicer(shard_envoy_pb2_grpc.ShardEnvoyServicer):

    def __init__(
        self, model_path: str, config_filename: str, conn: connection.Connection
    ) -> None:
        self.conn = conn
        self.config = self.get_config(model_path, config_filename)

        self.gen_queue = deque()
        self.buffer = Buffer(self.config["n_layers"], len(self.config["oth_urls"]))

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

    def sync_w_oths(
        self,
        executor: futures.Executor,
        oth_shards: dict,
        layer_num: int = None,
        before_warming: bool = False,
    ) -> None:
        if before_warming:
            layer_num = self.config["n_layers"]  # points to buffer's last bin
        else:
            self.conn.recv()  # means warmer is done

        fs = []
        for shard in oth_shards.values():
            fut = executor.submit(
                shard.SignalReady,
                shard_envoy_pb2.Identifier(layer_num=layer_num),
            )
            fs.append(fut)

        tic = time.perf_counter()
        futures.wait(fs)
        logging.info(f"signalling took {time.perf_counter() - tic} sec(s)")
        self.buffer.wait_until_full(layer_num)
        self.buffer.reset(layer_num)

    def all_dispatch(
        self, executor: futures.Executor, oth_shards: dict, fut_to_task: dict
    ) -> None:
        data = self.conn.recv_bytes()
        metadata = self.conn.recv_bytes()

        for shard in oth_shards.values():
            fut = executor.submit(
                shard.Receive,
                shard_envoy_pb2.ShardOuts(data=data, metadata=metadata),
            )
            fut_to_task[fut] = 0  # encoding for dispatch task

    def SignalReady(self, request: shard_envoy_pb2.Identifier, context):
        self.buffer.put(True, request.layer_num)
        return shard_envoy_pb2.Empty()

    def Receive(self, request: shard_envoy_pb2.ShardOuts, context):
        url, layer_num, bi, arr_map = pickle.loads(request.metadata)
        self.buffer.put((request.data, request.metadata), layer_num, bi)
        return shard_envoy_pb2.Empty()

    def Generate(self, request: shard_envoy_pb2.UsrIns, context):
        logging.info(f"received generation request")
        job = {"req": request}
        self.gen_queue.append(job)
        while "resp" not in job:
            pass
        return shard_envoy_pb2.UsrOuts(
            prompt_time=job["resp"][0],
            prompt_t_cnt=job["resp"][1],
            gen_time=job["resp"][2],
            gen_t_cnt=job["resp"][3],
            response=job["resp"][4],
        )

    def start(self) -> None:

        global DEFAULT_STARTUP_WARMING_PERIOD

        with ExitStack() as es:
            executor = es.enter_context(futures.ThreadPoolExecutor())
            oth_shards = {}
            for url in self.config["oth_urls"]:
                channel = es.enter_context(
                    grpc.insecure_channel(
                        url,
                        options=[
                            ("grpc.max_send_message_length", -1),
                            ("grpc.max_receive_message_length", -1),
                        ],
                    )
                )
                shard = shard_envoy_pb2_grpc.ShardEnvoyStub(channel)
                oth_shards[url] = shard

            while True:
                if len(self.gen_queue) == 0 or DEFAULT_STARTUP_WARMING_PERIOD > 0:
                    self.sync_w_oths(executor, oth_shards, before_warming=True)
                    logging.info(f"warming...")
                    self.conn.send(False)  # signal Generator that this is a warming run

                    for i in range(self.config["n_layers"]):
                        self.sync_w_oths(executor, oth_shards, layer_num=i)
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
                        fut_to_task = {}
                        for bi in range(batch_size):
                            self.all_dispatch(executor, oth_shards, fut_to_task)
                            fut_to_task[
                                executor.submit(self.buffer.wait_until_full, li, bi)
                            ] = 1

                        for fut in futures.as_completed(fut_to_task):
                            if fut_to_task[fut] == 0:
                                continue
                            for data, metadata in fut.result():
                                self.conn.send_bytes(data)
                                self.conn.send_bytes(metadata)

                        self.buffer.reset(li)

                    continue_sig = self.conn.recv()
                    if not continue_sig:
                        break

                gen["resp"] = self.conn.recv()
                self.gen_queue.popleft()


def envoy_main(
    port: int, model_path: str, config_filename: str, conn: connection.Connection
):
    logging.basicConfig(level=logging.INFO)

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_send_message_length", -1),
            ("grpc.max_receive_message_length", -1),
        ],
    )
    servicer = ShardEnvoyServicer(model_path, config_filename, conn)
    shard_envoy_pb2_grpc.add_ShardEnvoyServicer_to_server(servicer, server)
    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)
    server.start()
    logging.info(f"server started, listening on {listen_addr}")
    conn.recv()  # wait for Generator to finish initializing before starting to warm up
    servicer.start()

    # this might not be needed because servicer.start() enters an infinite loop
    server.wait_for_termination()
    conn.close()


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
