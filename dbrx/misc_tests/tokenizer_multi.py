#!/Users/xiangruike/miniconda3/envs/dbrx_poc/bin/python

from pathlib import Path

import mlx.core as mx

from transformers import AutoTokenizer


class Test:

    def __init__(self, model_path: str) -> None:
        self.model_path = Path(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )

    def test(self, prompts: list[str]):
        print(mx.array(self.tokenizer.encode(prompts))[None])
        # print(self.tokenizer(prompts))

if __name__ == "__main__":
    test = Test("/Users/xiangruike/dbrx-instruct/distributable/batch2")
    test.test("hello")
    # test.test(["hello there"]) 
