# Nemotron 3.5 Lightning on two nodes

These three files start **BF16 pretraining from random initialization** for `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` on two nodes with eight GPUs each. They use your real Megatron indexed data (`.bin` and `.idx`). The model architecture and tokenizer come from Megatron-Bridge's Lightning pretraining recipe. The published BF16 checkpoint weights are **not** loaded. For continued pretraining from those weights, a checkpoint import and a different initialization path are required.

The launcher follows the successful `nemo-pretrain.sbatch` pattern: Slurm starts one Pyxis container task per node, and each task runs `torchrun` with eight workers. The model settings follow NVIDIA's [Lightning BF16 pretraining recipe](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/recipes/nemotronh/h100/nemotron_3_nano.py) and [verified two-node H100 run](https://docs.nvidia.com/nemo/megatron-bridge/nightly/models/nemotron/nemotron3.5-lightning.html). Your [pinned Bridge reference](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/c4067f501aebefe03f27813f8ac0e4fe835249dd) is recorded here; its exact contents were unavailable from this environment, so the launcher checks that the Lightning recipe exists in the runtime image. The default image is `nvcr.io#nvidia/nemo:26.08`; the working 26.06 image may lack this recipe.

## Files

| File | Purpose |
| --- | --- |
| `lightning35.yaml` | Dataset, model parallelism, optimizer, step count, checkpoint interval |
| `lightning35.py` | Validates the dataset and applies YAML values to the Bridge Lightning recipe |
| `lightning35.sh` | Two-node Slurm/Pyxis/torchrun launcher; submit with `sbatch` |

## Prepare and submit

Run from a **shared directory accessible at the same path on both nodes**. Make sure Pyxis/Enroot works as in your successful job, and adjust `#SBATCH --partition=gpu` or the image path in `lightning35.sh` for your cluster. The example expects eight GPUs per node. NVIDIA's measured H100 run used this same node/GPU count; memory and timing on your hardware are unverified.

```bash
cd /shared/path/nemotron35_lightning_2nodes

# Example files:
# /shared/customdata/my_corpus_text_document.bin
# /shared/customdata/my_corpus_text_document.idx
# Supply the shared filename PREFIX, without .bin or .idx:
export DATA_PREFIX_HOST=/shared/customdata/my_corpus_text_document
test -s "${DATA_PREFIX_HOST}.bin" && test -s "${DATA_PREFIX_HOST}.idx"

# If your working setup uses a cached SquashFS image:
# export CONTAINER_IMAGE=/shared/images/nemo-26-08.sqsh

sbatch lightning35.sh
```

The launcher mounts the dataset directory read-only at `/data`. Thus the example becomes `/data/my_corpus_text_document` inside each container, and YAML's `${DATA_PREFIX}` resolves to that prefix. Both nodes check the files before GPU training starts. The `.bin` and `.idx` files must be generated with the **same tokenizer and token IDs** as the Lightning model; a file pair created with the Qwen tokenizer from the earlier successful run cannot be reused. See the [Megatron-Bridge data preparation guide](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/data-preparation.html).

The default YAML runs 20 optimizer steps at sequence length 8192, global batch 512 and micro batch 1. TP=1, PP=1, CP=2 and EP=8 produce eight dense data-parallel replicas across 16 GPUs. The dataset split is `100,0,0`; validation is disabled. Checkpoints are saved at steps 10 and 20 under `lightning35_runs/<job-id>/checkpoints`, and dataset indices plus TensorBoard files live under that same run directory. Check that this shared filesystem has enough free space: NVIDIA's verified run reported roughly **430 GiB per checkpoint**. Change step count, batch size, save interval, and optimizer values in `lightning35.yaml` for a longer run.

The run directory is new for each Slurm job. This bundle does not implement checkpoint resume. It has not been executed on your cluster; syntax and local configuration checks do not establish that its container, GPU memory, dataset size, or network will be sufficient.
