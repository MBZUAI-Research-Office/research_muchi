#!/Users/xiangruike/miniconda3/envs/dbrx_poc/bin/python

from pathlib import Path
import argparse
import gc
import glob
import json
import logging
import time

import safetensors.torch

# we found that mlx screws up data organization
# when an array is mx.split() before being mx.savez()
# pytorch is selected because it works with both bfloat16 and safetensors file format
import torch


class WeightsPreprocessor:

    def __init__(
        self,
        input_path: str,
        output_path: str,
        batching_strategy: int,
        skip_experts: bool,
        skip_sep: bool,
        clean_sep: bool,
    ) -> None:
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.batching_strategy = batching_strategy
        self.skip_experts = skip_experts
        self.skip_sep = skip_sep
        self.clean_sep = clean_sep

    def categorize_files(self, num_layers: int) -> dict:
        try:
            with open(self.input_path / "model.safetensors.index.json", "r") as f:
                metadata = json.load(f)
        except FileNotFoundError:
            logging.error(
                f"model.safetensors.index.json not found in {self.input_path}"
            )
            raise

        categorization = {
            "non_layer": set(),
            "layer": {i: set() for i in range(num_layers)},
        }
        for weight_name, file_name in metadata["weight_map"].items():
            if not weight_name.startswith("transformer.blocks"):
                categorization["non_layer"].add(file_name)
                continue

            layer_num = int(weight_name.split(".")[2])
            categorization["layer"][layer_num].add(file_name)

        return categorization

    def load_weights_in_map(self, weight_files: list, weight_map: set) -> dict:
        weights = {}
        for wf in weight_files:
            if Path(wf).name in weight_map:
                weights.update(safetensors.torch.load_file(wf))
        return weights

    def process_layer(
        self, layer_num: int, weights: dict, num_experts: int, ffn_hidden_size: int
    ) -> None:
        # a file can contain weights from multiple layers as well as non-layer weights
        layer_weights = {
            "attention_and_router": {},
            "experts": {i: {} for i in range(num_experts)},
        }
        for name, weight in weights.items():
            if not name.startswith("transformer.blocks"):
                continue

            name_split = name.split(".")

            if int(name_split[2]) != layer_num:
                continue

            if "experts.mlp" not in name:
                # len("transformer.") == 12, for our re-written model
                layer_weights["attention_and_router"][name[12:]] = weight
                continue

            # v1, w1, or w2
            linear_layer = name_split[-1]

            # split expert weights
            for i, expert in enumerate(torch.split(weight, ffn_hidden_size, dim=0)):
                layer_weights["experts"][i][
                    f"blocks.{layer_num}.experts.{i}.{linear_layer}.weight"
                ] = expert

        safetensors.torch.save_file(
            layer_weights["attention_and_router"],
            self.output_path / f"block{layer_num}-attention-and-router.safetensors",
        )
        for i in range(num_experts):
            safetensors.torch.save_file(
                layer_weights["experts"][i],
                self.output_path / f"block{layer_num}-expert{i}.safetensors",
            )

        print(f"processed layer {layer_num}")

    def process_non_layer(self, weights: dict) -> None:
        non_layer_weights = {}
        for k, v in weights.items():
            if not k.startswith("transformer.blocks"):
                # for our re-written model
                non_layer_weights[k[12:] if k.startswith("transformer.") else k] = v

        safetensors.torch.save_file(
            non_layer_weights, self.output_path / f"non-layer.safetensors"
        )

        print(f"processed non-layer weights")

    def clean_if_told(self, sep_paths: list[Path]) -> None:
        if self.clean_sep:
            for path in sep_paths:
                path.unlink()

    def batch_expert(self, expert_num: int, num_layers: int) -> None:
        sep_paths = []

        if self.batching_strategy == 1:
            expert_weights = {}
            for i in range(num_layers):
                path = self.input_path / f"block{i}-expert{expert_num}.safetensors"
                expert_weights.update(safetensors.torch.load_file(path))
                sep_paths.append(path)

            safetensors.torch.save_file(
                expert_weights, self.output_path / f"expert{expert_num}.safetensors"
            )
        elif self.batching_strategy == 2:
            expert_weights = []
            for i in range(num_layers):
                path = self.input_path / f"block{i}-expert{expert_num}.safetensors"
                weights = safetensors.torch.load_file(path)
                for j in ["v1", "w1", "w2"]:
                    k = f"blocks.{i}.experts.{expert_num}.{j}.weight"
                    expert_weights.append(weights[k])
                sep_paths.append(path)

            safetensors.torch.save_file(
                {"weights": torch.stack(expert_weights, dim=0)},
                self.output_path / f"expert{expert_num}.safetensors",
            )

        print(f"batched expert {expert_num} weights")
        self.clean_if_told(sep_paths)

    def batch_non_experts(self, num_layers: int) -> None:
        ne_weights = {}
        wqkvs, out_projs, routers = [], [], []
        sep_paths = []
        for i in range(num_layers):
            path = self.input_path / f"block{i}-attention-and-router.safetensors"
            sep_paths.append(path)
            weights = safetensors.torch.load_file(path)
            wqkvs.append(weights.pop(f"blocks.{i}.norm_attn_norm.attn.Wqkv.weight"))
            out_projs.append(
                weights.pop(f"blocks.{i}.norm_attn_norm.attn.out_proj.weight")
            )
            routers.append(weights.pop(f"blocks.{i}.ffn.router.layer.weight"))
            ne_weights.update(weights)  # norm_1 & norm_2

        non_layer_path = self.input_path / f"non-layer.safetensors"
        ne_weights.update(safetensors.torch.load_file(non_layer_path))
        sep_paths.append(non_layer_path)

        ne_weights["wqkv_weights"] = torch.stack(wqkvs, dim=0)
        ne_weights["out_proj_weights"] = torch.stack(out_projs, dim=0)
        ne_weights["router_weights"] = torch.stack(routers, dim=0)

        safetensors.torch.save_file(
            ne_weights, self.output_path / f"non-expert.safetensors"
        )

        print(f"batched non-expert weights")
        self.clean_if_told(sep_paths)

    def sep_weights_by_layer_and_expert(
        self, num_layers: int, num_experts: int, ffn_hidden_size: int
    ):
        tic = time.perf_counter()

        categorization = self.categorize_files(num_layers)
        weight_files = glob.glob(str(self.input_path / "*.safetensors"))
        if not weight_files:
            logging.error(f"No safetensors found in {self.input_path}")
            raise FileNotFoundError(f"No safetensors found in {self.input_path}")

        for i in categorization["layer"]:
            self.process_layer(
                i,
                self.load_weights_in_map(weight_files, categorization["layer"][i]),
                num_experts,
                ffn_hidden_size,
            )
            gc.collect()

        self.process_non_layer(
            self.load_weights_in_map(weight_files, categorization["non_layer"])
        )
        gc.collect()

        # in case batching is requested:
        # batching function's input comes from this function's output
        self.input_path = self.output_path

        print(
            "weights re-organized by layer and expert in "
            + f"{time.perf_counter() - tic} sec(s)"
        )

    def batch_weights(self, num_layers: int, num_experts: int):
        tic = time.perf_counter()

        if not self.skip_experts:
            for i in range(num_experts):
                self.batch_expert(i, num_layers)
                gc.collect()

        self.batch_non_experts(num_layers)
        gc.collect()

        print(f"weights batched in {time.perf_counter() - tic} sec(s)")

    def start(self) -> None:
        try:
            with open(self.input_path / "config.json", "r") as f:
                config = json.load(f)
        except FileNotFoundError:
            logging.error(f"Config file not found in {self.input_path}")
            raise

        print("=" * 20)
        print("WEIGHTS PREPROCESSOR STARTED")

        num_layers = config["n_layers"]
        num_experts = config["ffn_config"]["moe_num_experts"]

        if not self.skip_sep:
            self.sep_weights_by_layer_and_expert(
                num_layers, num_experts, config["ffn_config"]["ffn_hidden_size"]
            )

        if self.batching_strategy > 0:
            self.batch_weights(num_layers, num_experts)

        print("=" * 20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str)
    parser.add_argument("--output-path", type=str)
    parser.add_argument(
        "--batching-strategy",
        type=int,
        default=0,
        help="0: no batching, 1: batch by expert, 2: batch by and within expert",
    )
    parser.add_argument("--skip-experts", action="store_true")
    parser.add_argument("--skip-sep", action="store_true")
    parser.add_argument("--clean-sep", action="store_true")
    args = parser.parse_args()
    logging.basicConfig()

    weights_preprocessor = WeightsPreprocessor(
        args.input_path,
        args.output_path,
        args.batching_strategy,
        args.skip_experts,
        args.skip_sep,
        args.clean_sep,
    )
    weights_preprocessor.start()
