#!/usr/bin/env python3
"""Two-node Nemotron 3.5 Lightning pretraining on Megatron indexed data."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


EXPECTED_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"


def load_settings(config_path: Path) -> dict:
    with config_path.open(encoding="utf-8") as stream:
        settings = yaml.safe_load(stream)
    if not isinstance(settings, dict):
        raise ValueError("Configuration must be a YAML mapping")
    required = {"model", "dataset", "training", "optimizer", "checkpoint", "logging"}
    if set(settings) != required:
        raise ValueError(f"Expected YAML sections: {sorted(required)}")
    return settings


def validate(settings: dict, check_files: bool = True) -> str:
    model = settings["model"]
    data = settings["dataset"]
    train = settings["training"]
    optim = settings["optimizer"]
    ckpt = settings["checkpoint"]

    if model["id"] != EXPECTED_MODEL:
        raise ValueError(f"This recipe expects {EXPECTED_MODEL}")
    if (model["tensor_parallel"], model["pipeline_parallel"],
            model["context_parallel"], model["expert_parallel"]) != (1, 1, 2, 8):
        raise ValueError("This two-node recipe requires TP=1, PP=1, CP=2, EP=8")
    if model["seq_length"] != 8192:
        raise ValueError("This Lightning recipe is configured for sequence length 8192")
    if data["split"] != "100,0,0":
        raise ValueError("Validation is disabled here; dataset.split must be 100,0,0")

    prefix = os.path.expandvars(data["prefix"])
    if not prefix or "$" in prefix or not Path(prefix).is_absolute():
        raise ValueError("Set DATA_PREFIX_HOST to an absolute .bin/.idx filename prefix")
    if prefix.endswith((".bin", ".idx")):
        raise ValueError("Dataset prefix must omit the .bin and .idx extensions")
    if check_files:
        for extension in (".bin", ".idx"):
            candidate = Path(prefix + extension)
            if not candidate.is_file() or candidate.stat().st_size == 0:
                raise FileNotFoundError(f"Dataset file missing or empty: {candidate}")

    for name in ("iterations", "micro_batch_size", "global_batch_size", "seed"):
        if type(train[name]) is not int or train[name] < (0 if name == "seed" else 1):
            raise ValueError(f"training.{name} must be a positive integer")
    # 16 ranks / CP=2 = 8 dense data-parallel replicas.
    if train["global_batch_size"] % (8 * train["micro_batch_size"]):
        raise ValueError("Global batch size must be divisible by 8 * micro batch size")
    if type(data["num_workers"]) is not int or data["num_workers"] < 0:
        raise ValueError("dataset.num_workers must be nonnegative")
    if not (0 < optim["min_lr"] <= optim["lr"]):
        raise ValueError("Require 0 < optimizer.min_lr <= optimizer.lr")
    if not (0 <= optim["warmup_iters"] < train["iterations"]):
        raise ValueError("Warmup must be shorter than training")
    if type(ckpt["save_interval"]) is not int or ckpt["save_interval"] < 1:
        raise ValueError("checkpoint.save_interval must be positive")
    if type(settings["logging"]["log_interval"]) is not int or settings["logging"]["log_interval"] < 1:
        raise ValueError("logging.log_interval must be positive")
    return prefix


def build_config(settings: dict, prefix: str):
    from megatron.bridge.recipes.nemotronh.nemotron_3_nano import (
        nemotron_3_5_lightning_pretrain_config,
    )

    cfg = nemotron_3_5_lightning_pretrain_config()
    model = settings["model"]
    train = settings["training"]
    optim = settings["optimizer"]
    output = Path(os.environ["RUN_OUTPUT_DIR"])

    if cfg.model.hf_model_id != EXPECTED_MODEL or cfg.model.mtp_num_layers != 2:
        raise RuntimeError("Installed Megatron-Bridge has the wrong Lightning recipe; use the supported Bridge version")
    if cfg.tokenizer.tokenizer_model != EXPECTED_MODEL:
        raise RuntimeError("Lightning tokenizer does not match the selected model")

    cfg.model.seq_length = model["seq_length"]
    cfg.model.tensor_model_parallel_size = model["tensor_parallel"]
    cfg.model.pipeline_model_parallel_size = model["pipeline_parallel"]
    cfg.model.context_parallel_size = model["context_parallel"]
    cfg.model.expert_model_parallel_size = model["expert_parallel"]
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.cp_comm_type = "p2p"
    cfg.model.moe_router_force_load_balancing = False
    # All-to-all avoids requiring a site-specific HybridEP/DeepEP setup.
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_flex_dispatcher_backend = None
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.cuda_graph_impl = "none"

    cfg.dataset.blend = ([prefix], None)
    cfg.dataset.blend_per_split = None
    cfg.dataset.mock = False
    cfg.dataset.split = settings["dataset"]["split"]
    cfg.dataset.seq_length = model["seq_length"]
    cfg.dataset.num_workers = settings["dataset"]["num_workers"]
    cfg.dataset.path_to_cache = str(output / "dataset_cache")
    cfg.dataset.random_seed = train["seed"]

    cfg.train.train_iters = train["iterations"]
    cfg.train.micro_batch_size = train["micro_batch_size"]
    cfg.train.global_batch_size = train["global_batch_size"]
    cfg.rng.seed = train["seed"]
    cfg.optimizer.lr = optim["lr"]
    cfg.optimizer.min_lr = optim["min_lr"]
    cfg.scheduler.lr_warmup_iters = optim["warmup_iters"]
    cfg.scheduler.lr_decay_iters = train["iterations"]
    cfg.scheduler.lr_decay_style = "cosine"
    cfg.validation.eval_interval = 0
    cfg.validation.eval_iters = 0
    cfg.checkpoint.load = None
    cfg.checkpoint.pretrained_checkpoint = None
    cfg.checkpoint.finetune = False
    cfg.checkpoint.save = str(output / "checkpoints")
    cfg.checkpoint.save_interval = settings["checkpoint"]["save_interval"]
    cfg.checkpoint.async_save = False
    cfg.logger.log_interval = settings["logging"]["log_interval"]
    cfg.logger.log_throughput = True
    cfg.logger.tensorboard_dir = str(output / "tensorboard")
    cfg.dist.distributed_timeout_minutes = 120
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_suffix(".yaml"))
    parser.add_argument("--check", action="store_true", help="check config and dataset without starting training")
    args = parser.parse_args()
    settings = load_settings(args.config)
    prefix = validate(settings)
    if args.check:
        print(f"Dataset ready on {os.uname().nodename}: {prefix}.bin / {prefix}.idx", flush=True)
        return

    if int(os.environ.get("WORLD_SIZE", "0")) != 16:
        raise RuntimeError("Expected 2 nodes x 8 GPUs = 16 torchrun workers")
    if "RUN_OUTPUT_DIR" not in os.environ:
        raise RuntimeError("RUN_OUTPUT_DIR is missing; launch through lightning35.sh")

    from megatron.bridge.training.gpt_step import forward_step
    from megatron.bridge.training.pretrain import pretrain

    cfg = build_config(settings, prefix)
    if int(os.environ.get("RANK", "-1")) == 0:
        print(f"Lightning BF16 pretraining: 16 GPUs, real indexed data at {prefix}", flush=True)
        print(f"Output: {os.environ['RUN_OUTPUT_DIR']}", flush=True)
    pretrain(cfg, forward_step)


if __name__ == "__main__":
    main()
