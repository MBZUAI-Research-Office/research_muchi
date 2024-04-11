#!/Users/xiangruike/miniconda3/envs/dbrx_poc/bin/python

from pathlib import Path
import argparse
import gc
import glob
import json
import logging

# for DEV
import pprint

import mlx.core as mx


class WeightsPreprocessor:

    def __init__(self, model_path: str, output_dir: str) -> None:
        self.model_path = Path(model_path)
        self.output_dir = self.model_path / output_dir

    def categorize_files(self) -> dict:
        try:
            with open(self.model_path / "model.safetensors.index.json", "r") as f:
                metadata = json.load(f)
        except FileNotFoundError:
            logging.error(
                f"model.safetensors.index.json not found in {self.model_path}"
            )
            raise

        categorization = {"non_layer": set(), "layer": {}}
        for weight_name, file_name in metadata["weight_map"].items():
            if not weight_name.startswith("transformer.blocks"):
                categorization["non_layer"].add(file_name)
                continue

            layer_num = weight_name.split(".")[2]  # dtype=str
            categorization["layer"].setdefault(layer_num, set()).add(file_name)

        return categorization

    def load_weights_in_map(self, weight_files: list, weight_map: set) -> dict:
        weights = {}
        for wf in weight_files:
            if Path(wf).name in weight_map:
                weights.update(mx.load(wf))
        return weights

    def process_layer(self, layer_num: str, weights: dict, num_experts: int):
        # a file can contain weights from multiple layers as well as non-layer weights
        layer_weights = {
            "attention_and_router": {},
            "experts": {i: {} for i in range(num_experts)},
        }
        for name, weight in weights.items():
            if not name.startswith("transformer.blocks"):
                continue

            name_split = name.split(".")

            if name_split[2] != layer_num:
                continue

            if "experts.mlp" not in name:
                layer_weights["attention_and_router"][name] = weight
                continue

            # split expert weights
            linear_layer_name = name_split[-1]
            for i, expert in enumerate(mx.split(weight, num_experts, axis=0)):
                layer_weights["experts"][i][
                    f"experts.{i}.{linear_layer_name}.weight"
                ] = (expert.T if linear_layer_name == "w2" else expert)

        mx.savez(
            self.output_dir / f"block{layer_num}-attention-and-router.npz",
            **layer_weights["attention_and_router"],
        )
        for i in range(num_experts):
            mx.savez(
                self.output_dir / f"block{layer_num}-expert{i}.npz",
                **layer_weights["experts"][i],
            )
        print(f"processed layer {layer_num}")

        # forces python to free up memory
        del weights
        del layer_weights
        gc.collect()

    def process_non_layer(self, weights: dict):
        non_layer_weights = {
            k: v for k, v in weights.items() if not k.startswith("transformer.blocks")
        }
        mx.savez(self.output_dir / f"non-layer.npz", **non_layer_weights)
        print(f"processed non-layer weights")

        # forces python to free up memory
        del weights
        del non_layer_weights
        gc.collect()

    def start(self):
        try:
            with open(self.model_path / "config.json", "r") as f:
                config = json.load(f)
        except FileNotFoundError:
            logging.error(f"Config file not found in {self.model_path}")
            raise

        weight_files = glob.glob(str(self.model_path / "*.safetensors"))
        if not weight_files:
            logging.error(f"No safetensors found in {self.model_path}")
            raise FileNotFoundError(f"No safetensors found in {self.model_path}")

        categorization = self.categorize_files()
        for i in categorization["layer"]:
            self.process_layer(
                i,
                self.load_weights_in_map(weight_files, categorization["layer"][i]),
                config["ffn_config"]["moe_num_experts"],
            )
        self.process_non_layer(
            self.load_weights_in_map(weight_files, categorization["non_layer"])
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--output-dir", type=str)
    args = parser.parse_args()
    logging.basicConfig()
    weights_processor = WeightsPreprocessor(args.model_path, args.output_dir)
    weights_processor.start()
