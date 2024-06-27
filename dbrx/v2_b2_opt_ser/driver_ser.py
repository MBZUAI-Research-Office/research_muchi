#!/Users/xiangruike/miniconda3/envs/dbrx_poc/bin/python

from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import argparse
import asyncio
import inspect
import json
import logging
import time

import grpc
import moe_shard_ser_pb2
import moe_shard_ser_pb2_grpc

import mlx.core as mx
import mlx.nn as nn

from transformers import AutoTokenizer, PreTrainedTokenizer

from serialization_utils import mx_to_bytes, bytes_to_mx

DEFAULT_PROMPT = "hello"
DEFAULT_MAX_TOKENS = 100
DEFAULT_TEMP = 0.6

import statistics
import pprint

STATS = {
    "moe_lat": [],
    "comm_lat": [],
    "misc_lat": [],
    "prompt_eval_tp": [],
    "token_gen_tp": [],
}
LOGS = {"moe_lat": [], "comm_lat": []}


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


class DistributedSparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.d_model = args.d_model
        self.num_experts_per_tok = args.ffn_config["moe_top_k"]
        self.expert_to_url = args.ffn_config["expert_to_url"]
        self.moe_shard_map = args.ffn_config["moe_shard_map"]
        self.router = Router(args.d_model, args.ffn_config["moe_num_experts"])

    async def execute_on_shard(
        self,
        shard: moe_shard_ser_pb2_grpc.MoeShardStub,
        x_bytes: bytes,  # x.shape == (batch_size, self.d_model)
    ):
        outputs = await shard.Execute(moe_shard_ser_pb2.Inputs(data=x_bytes))
        return bytes_to_mx(outputs.data), outputs.exec_time

    async def __call__(self, x: mx.array) -> mx.array:
        ne = self.num_experts_per_tok
        orig_shape = x.shape
        x = x.reshape(-1, x.shape[-1])

        gates = self.router(x)
        gates = mx.softmax(gates.astype(mx.float32), axis=-1)

        inds = mx.stop_gradient(mx.argpartition(-gates, kth=ne, axis=-1)[:, :ne])
        scores = mx.take_along_axis(gates, inds, axis=-1)
        scores = scores / mx.linalg.norm(scores, ord=1, axis=-1, keepdims=True)
        scores = scores.astype(x.dtype)

        mx.eval(inds, scores)  # fucking magic: from 2.251 t/s to 2.988 t/s
        x_bytes = mx_to_bytes(x)
        y = []
        batch_size = x.shape[0]

        tic = time.perf_counter_ns()

        async with asyncio.TaskGroup() as tg:
            exec_tasks = {}
            for url, d in self.moe_shard_map.items():
                task = tg.create_task(self.execute_on_shard(d["shard"], x_bytes))
                exec_tasks[url] = task

        toc = time.perf_counter_ns()
        expert_latency = statistics.mean(
            task.result()[1] for task in exec_tasks.values()
        )
        LOGS["moe_lat"].append(expert_latency)
        LOGS["comm_lat"].append((toc - tic) - expert_latency)

        for bi, st, it in zip(range(batch_size), scores, inds.tolist()):
            yt = []
            for e in it:
                url = self.expert_to_url[e]
                ei = self.moe_shard_map[url]["expert_to_i"][e]
                res = exec_tasks[url].result()[0][bi, ei]
                yt.append(res)

            yt = mx.stack(yt, axis=-1)
            yt = (yt * st).sum(axis=-1)
            y.append(yt)

        y = mx.stack(y, axis=0)
        return y.reshape(orig_shape)


class DistributedDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.ffn = DistributedSparseMoeBlock(args)
        self.norm_attn_norm = NormAttnNorm(args)

    async def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        r, h, cache = self.norm_attn_norm(x, mask, cache)
        out = (await self.ffn(h)) + r
        return out, cache


class DistributedDBRX(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.wte = nn.Embedding(args.vocab_size, args.d_model)
        self.blocks = [DistributedDecoderLayer(args) for _ in range(args.n_layers)]
        self.norm_f = nn.LayerNorm(args.d_model, bias=False)
        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)

    async def __call__(
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

        for e, layer in enumerate(self.blocks):
            h, cache[e] = await layer(h, mask, cache[e])

        return self.lm_head(self.norm_f(h)), cache


def get_json(file_path: Path) -> dict:
    try:
        with open(file_path, "r") as f:
            res = json.load(f)
    except FileNotFoundError:
        logging.error(f"{file_path} not found")
        raise

    return res


class Driver:
    def __init__(self, model_path: str) -> None:
        self.model_path = Path(model_path)

    def get_model_args(self) -> ModelArgs:
        config = get_json(self.model_path / "v0_driver_config.json")
        model_args = ModelArgs.from_dict(config)
        model_args.ffn_config["expert_to_url"] = {}

        for url in list(model_args.ffn_config["moe_shard_map"].keys()):
            assigned_experts = model_args.ffn_config["moe_shard_map"][url]
            expert_to_i = {}
            for i, e in enumerate(assigned_experts):
                model_args.ffn_config["expert_to_url"][e] = url
                expert_to_i[e] = i
            model_args.ffn_config["moe_shard_map"][url] = {"expert_to_i": expert_to_i}

        return model_args

    async def generate_step(
        self,
        model: nn.Module,
        prompt: mx.array,
        temp: float,
    ) -> AsyncGenerator[Tuple[mx.array, mx.array]]:

        def sample(logits: mx.array) -> Tuple[mx.array, float]:
            softmax_logits = mx.softmax(logits)

            if temp == 0:
                token = mx.argmax(logits, axis=-1)
            else:
                token = mx.random.categorical(logits * (1 / temp))

            prob = softmax_logits[0, token]
            return token, prob

        y = prompt
        cache = None

        while True:
            logits, cache = await model(y[None], cache=cache)
            logits = logits[:, -1, :]
            y, prob = sample(logits)
            yield y, prob

    async def generate(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        prompt: str,
        max_tokens: int,
        temp: float,
    ) -> str:
        print("=" * 10)
        print("Prompt:", prompt)

        prompt_tokens = mx.array(tokenizer.encode(prompt))

        tic = time.perf_counter()
        tokens = []
        token_strings = []
        skip = 0
        REPLACEMENT_CHAR = "\ufffd"

        n = 0
        async for token, prob in self.generate_step(
            model,
            prompt_tokens,
            temp,
        ):
            if n >= max_tokens:
                break

            token = token.item()  # get word ID
            if n == 0:
                prompt_time = time.perf_counter() - tic
                tic = time.perf_counter()
            if token == tokenizer.eos_token_id:
                n += 1
                break
            tokens.append(token)

            s = tokenizer.decode(tokens)  # str
            if s[-1] != REPLACEMENT_CHAR:
                # DEV
                print(s[skip:], end="", flush=True)
                # print("-" * 20, flush=True)

                skip = len(s)
            # Reset token cache at line break
            if s[-1] == "\n":
                tokens = []
                token_strings.append(s)
                skip = 0

            n += 1

        token_strings.append(tokenizer.decode(tokens).replace(REPLACEMENT_CHAR, ""))

        print(token_strings[-1][skip:], flush=True)
        gen_time = time.perf_counter() - tic
        prompt_eval_tp = prompt_tokens.size / prompt_time
        token_gen_tp = (n - 1) / gen_time
        print("=" * 10)
        if n == 0:
            print("No tokens generated for this prompt")
            return
        print(
            f"Prompt: {prompt_tokens.size} tokens in {prompt_time} seconds "
            + f"= {prompt_eval_tp:.3f} t/s"
        )
        print(
            f"Generation: {n - 1} tokens in {gen_time} seconds "
            + f"= {token_gen_tp:.3f} t/s"
        )

        if n >= max_tokens * 0.85:
            avg_moe_lat = statistics.mean(LOGS["moe_lat"][40:]) / (1000**2)
            avg_comm_lat = statistics.mean(LOGS["comm_lat"][40:]) / (1000**2)
            avg_misc_lat = (1000 / token_gen_tp / 40) - avg_moe_lat - avg_comm_lat
            STATS["moe_lat"].append(avg_moe_lat)
            STATS["comm_lat"].append(avg_comm_lat)
            STATS["misc_lat"].append(avg_misc_lat)
            STATS["prompt_eval_tp"].append(prompt_eval_tp)
            STATS["token_gen_tp"].append(token_gen_tp)
            return True

        LOGS["moe_lat"] = []
        LOGS["comm_lat"] = []

        return False

    async def start(
        self,
        prompt: str,
        prompt_path: str,
        n_samples: int,
        max_tokens: int,
        temp: float,
    ) -> None:
        async with AsyncExitStack() as es:
            model_args = self.get_model_args()
            for url in list(model_args.ffn_config["moe_shard_map"].keys()):
                channel = await es.enter_async_context(grpc.aio.insecure_channel(url))
                shard = moe_shard_ser_pb2_grpc.MoeShardStub(channel)
                model_args.ffn_config["moe_shard_map"][url]["shard"] = shard

            model = DistributedDBRX(model_args)
            weights = mx.load(str(self.model_path / f"non-expert.safetensors"))
            model.load_weights(list(weights.items()))
            mx.eval(model.parameters())
            model.eval()

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            prompts = get_json(Path(prompt_path))["prompts"] if not prompt else [prompt]
            n_satisfying_resp = 0

            for _ in range(n_samples):
                for p in prompts:
                    satisfied = await self.generate(model, tokenizer, p, max_tokens, temp)
                    n_satisfying_resp += int(satisfied)

            print(f"\nnumber of responses reaching max-tokens: {n_satisfying_resp}")
            print(f"AVG MoE latency: {statistics.mean(STATS['moe_lat'])} ms")
            print(f"AVG Comm latency: {statistics.mean(STATS['comm_lat'])} ms")
            print(f"AVG Misc latency: {statistics.mean(STATS['misc_lat'])} ms")
            print(f"AVG Prompt Eval TP: {statistics.mean(STATS['prompt_eval_tp'])} t/s")
            print(f"AVG Token Gen TP: {statistics.mean(STATS['token_gen_tp'])} t/s")
            pprint.pp(STATS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, help="Path to local model directory")
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
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
    parser.add_argument(
        "--temp", type=float, default=DEFAULT_TEMP, help="Sampling temperature"
    )
    args = parser.parse_args()

    # DEBUG
    mx.random.seed(0)

    logging.basicConfig()
    driver = Driver(args.model_path)
    asyncio.run(
        driver.start(
            args.prompt, args.prompt_path, args.n_samples, args.max_tokens, args.temp
        )
    )
