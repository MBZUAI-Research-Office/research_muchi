#!/Users/xiangruike/miniconda3/envs/mlperf/python
import argparse

import mlx.core as mx
from mlx_lm import load, generate


def main(args):
    # editted from:
    # https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/generate.py

    default_seed = 0
    default_temp = 0.6
    mx.random.seed(default_seed)

    model, tokenizer = load(args.model)

    if (
        hasattr(tokenizer, "apply_chat_template")
        and tokenizer.chat_template is not None
    ):
        messages = [{"role": "user", "content": args.prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt = args.prompt

    generate(model, tokenizer, prompt, temp=default_temp, verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="runs llama 2 with mlx_lm")
    parser.add_argument(
        "--model",
        type=str,
        help="path to self hf model stored locally"
        + "see: https://huggingface.co/meta-llama",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="hello",
        help="input to the specified model",
    )
    args = parser.parse_args()

    main(args)
