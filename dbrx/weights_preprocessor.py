#!/Users/xiangruike/miniconda3/envs/dbrx_poc/bin/python

from pathlib import Path
import argparse
import gc
import glob
import json
import logging
import time

import mlx.core as mx


class WeightsPreprocessor:

    def __init__(
        self, model_path: str, output_dir: str, batching_strategy: int, clean_sep: bool
    ) -> None:
        self.model_path = Path(model_path)
        self.output_dir = self.model_path / output_dir
        self.batching_strategy = batching_strategy
        self.clean_sep = clean_sep

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

    def process_layer(self, layer_num: str, weights: dict, num_experts: int) -> None:
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
                # len("transformer.") == 12, for our re-written model
                layer_weights["attention_and_router"][name[12:]] = weight
                continue

            # split expert weights
            linear_layer_name = name_split[-1]
            for i, expert in enumerate(mx.split(weight, num_experts, axis=0)):
                # TODO: original safetensors omit .weights
                layer_weights["experts"][i][
                    f"blocks.{layer_num}.experts.{i}.{linear_layer_name}.weight"
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

    def process_non_layer(self, weights: dict) -> None:
        non_layer_weights = {}
        for k, v in weights.items():
            if not k.startswith("transformer.blocks"):
                # for our re-written model
                non_layer_weights[k[12:] if k.startswith("transformer.") else k] = v

        mx.savez(self.output_dir / f"non-layer.npz", **non_layer_weights)
        print(f"processed non-layer weights")

        # forces python to free up memory
        del weights
        del non_layer_weights
        gc.collect()

    def clean_if_told(self, sep_paths: list[Path]) -> None:
        if self.clean_sep:
            for path in sep_paths:
                path.unlink()

    def batch_experts(self, num_layers: int, num_experts: int) -> None:
        for i in range(num_experts):
            expert_weights = {}
            sep_paths = []
            for j in range(num_layers):
                path = self.output_dir / f"block{j}-expert{i}.npz"
                expert_weights.update(mx.load(str(path)))
                sep_paths.append(path)

            if self.batching_strategy == 1:
                mx.savez(self.output_dir / f"expert{i}.npz", **expert_weights)
            elif self.batching_strategy == 2:
                mx.savez(
                    self.output_dir / f"expert{i}.npz",
                    layer1=mx.stack(
                        [
                            expert_weights[f"blocks.{j}.experts.{i}.{k}.weight"]
                            for j in range(num_layers)
                            for k in ["v1", "w1"]
                        ],
                        axis=0,
                    ),
                    layer2=mx.stack(
                        [
                            expert_weights[f"blocks.{j}.experts.{i}.w2.weight"]
                            for j in range(num_layers)
                        ],
                        axis=0,
                    ),
                )

            print(f"batched expert {i} weights")

            self.clean_if_told(sep_paths)

            # forces python to free up memory
            del expert_weights
            del sep_paths
            gc.collect()

    def batch_non_experts(self, num_layers: int) -> None:
        non_expert_weights = {}
        sep_paths = []
        for i in range(num_layers):
            path = self.output_dir / f"block{i}-attention-and-router.npz"
            non_expert_weights.update(mx.load(str(path)))
            sep_paths.append(path)

        non_layer_path = self.output_dir / f"non-layer.npz"
        non_expert_weights.update(mx.load(str(non_layer_path)))
        sep_paths.append(non_layer_path)

        mx.savez(self.output_dir / f"non-expert.npz", **non_expert_weights)
        # if self.batching_strategy == 1:
        #     mx.savez(self.output_dir / f"non-expert.npz", **non_expert_weights)
        # elif self.batching_strategy == 2:
        #     attention_and_router_order = [
        #         "ffn.router.layer",
        #         "norm_attn_norm.attn.Wqkv",
        #         "norm_attn_norm.attn.out_proj",
        #         "norm_attn_norm.norm_1",
        #         "norm_attn_norm.norm_2",
        #     ]
        #     sorted_k = (
        #         ["lm_head.weight"]
        #         + [
        #             f"blocks.{i}.{j}.weight"
        #             for i in range(num_layers)
        #             for j in attention_and_router_order
        #         ]
        #         + ["norm_f.weight", "wte.weight"]
        #     )
        #     mx.savez(
        #         self.output_dir / f"non-expert.npz",
        #         stacked_weights=mx.stack(
        #             [non_expert_weights[k] for k in sorted_k], axis=0
        #         ),
        #     )
        #     del attention_and_router_order
        #     del sorted_k

        print(f"batched non-expert weights")

        self.clean_if_told(sep_paths)

        # forces python to free up memory
        del non_expert_weights
        del sep_paths
        del non_layer_path
        gc.collect()

    def start(self) -> None:
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

        print("=" * 20)
        print("WEIGHTS PREPROCESSOR STARTED")
        tic = time.perf_counter()

        num_layers = config["n_layers"]
        num_experts = config["ffn_config"]["moe_num_experts"]
        categorization = self.categorize_files()
        for i in categorization["layer"]:
            self.process_layer(
                i,
                self.load_weights_in_map(weight_files, categorization["layer"][i]),
                num_experts,
            )
        self.process_non_layer(
            self.load_weights_in_map(weight_files, categorization["non_layer"])
        )

        print(
            "weights re-organized by layer and expert in "
            + f"{time.perf_counter() - tic} sec(s)"
        )

        if self.batching_strategy > 0:
            tic = time.perf_counter()

            self.batch_experts(num_layers, num_experts)
            self.batch_non_experts(num_layers)

            print(f"weights batched in {time.perf_counter() - tic} sec(s)")

        print("=" * 20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument(
        "--batching-strategy",
        type=int,
        default=0,
        help="0: no batching, 1: batch by expert, 2: batch by and within expert",
    )
    parser.add_argument("--clean-sep", action="store_true")
    args = parser.parse_args()
    logging.basicConfig()

    weights_preprocessor = WeightsPreprocessor(
        args.model_path, args.output_dir, args.batching_strategy, args.clean_sep
    )
    weights_preprocessor.start()
