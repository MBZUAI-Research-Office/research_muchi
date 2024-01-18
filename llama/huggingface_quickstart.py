import argparse
import time

from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
import transformers
import torch


class Llama2:
    def __init__(
        self, local_model_path: str = None, remote_model_name: str = None
    ) -> None:
        if local_model_path is not None:
            # if both args are specified, the local model is used
            self.model = LlamaForCausalLM.from_pretrained(local_model_path)
            self.tokenizer = LlamaTokenizer.from_pretrained(local_model_path)
        else:
            self.model = remote_model_name
            self.tokenizer = AutoTokenizer.from_pretrained(remote_model_name)

    def run_inference(self, prompt: str) -> None:
        start_time = time.time()
        pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        sequences = pipeline(
            prompt + "\n",
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=200,
        )

        for seq in sequences:
            print(f"Result: {seq['generated_text']}")

        print(f"--- inference completed in {time.time() - start_time} seconds ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="runs llama 2 with huggingface")
    parser.add_argument(
        "--local-model-path",
        type=str,
        help="path to self converted model stored locally"
        + "see: https://huggingface.co/docs/transformers/model_doc/llama2",
    )
    parser.add_argument(
        "--remote-model-name",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",  # just in case local model path is not specified
        help="name of readily converted models on hf"
        + "see: https://huggingface.co/meta-llama",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="input to the specified model",
    )
    args = parser.parse_args()

    model = Llama2(
        local_model_path=args.local_model_path, remote_model_name=args.remote_model_name
    )
    model.run_inference(args.prompt)
