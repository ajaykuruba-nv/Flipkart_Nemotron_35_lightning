#!/usr/bin/env bash
#SBATCH --job-name=lightning35-pretrain
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Submit from the directory containing lightning35.py and lightning35.yaml.
# DATA_PREFIX_HOST must name the shared .bin/.idx prefix, without an extension.
set -Eeuo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
: "${DATA_PREFIX_HOST:?Export DATA_PREFIX_HOST=/shared/path/corpus_text_document before sbatch}"
[[ "$DATA_PREFIX_HOST" == /* && "$DATA_PREFIX_HOST" != *.bin && "$DATA_PREFIX_HOST" != *.idx ]] || {
    echo "DATA_PREFIX_HOST must be an absolute prefix without .bin/.idx" >&2
    exit 2
}
[[ -f "${SUBMIT_DIR}/lightning35.py" && -f "${SUBMIT_DIR}/lightning35.yaml" ]] || {
    echo "Submit from the directory containing lightning35.py and lightning35.yaml" >&2
    exit 2
}
[[ "${SLURM_NNODES:?}" == 2 ]] || { echo "Exactly two nodes are required" >&2; exit 2; }

DATA_DIR_HOST="$(dirname "$DATA_PREFIX_HOST")"
for extension in bin idx; do
    [[ -s "${DATA_PREFIX_HOST}.${extension}" ]] || {
        echo "Missing or empty: ${DATA_PREFIX_HOST}.${extension}" >&2
        exit 2
    }
done

CONTAINER_IMAGE="${CONTAINER_IMAGE:-nvcr.io#nvidia/nemo:26.08}"
CACHE_DIR="${SUBMIT_DIR}/.cache"
RUN_OUTPUT_DIR="${SUBMIT_DIR}/lightning35_runs/${SLURM_JOB_ID}"
mkdir -p "$CACHE_DIR/enroot" "$RUN_OUTPUT_DIR"
export ENROOT_CACHE_PATH="${CACHE_DIR}/enroot"
export SUBMIT_DIR RUN_OUTPUT_DIR
export DATA_PREFIX="/data/$(basename "$DATA_PREFIX_HOST")"
export NNODES=2
export MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
export MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONUNBUFFERED=1

echo "Job ${SLURM_JOB_ID}: 2 nodes x 8 GPUs; master ${MASTER_ADDR}:${MASTER_PORT}"
echo "Dataset: ${DATA_PREFIX_HOST}.{bin,idx} -> ${DATA_PREFIX}.{bin,idx}"
echo "Image: ${CONTAINER_IMAGE}"
echo "Output: ${RUN_OUTPUT_DIR}"

# One Slurm task on each node; each task starts eight local torchrun workers.
srun \
    --nodes=2 --ntasks=2 --ntasks-per-node=1 --kill-on-bad-exit=1 \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${SUBMIT_DIR}:${SUBMIT_DIR},${CACHE_DIR}:/root/.cache,${DATA_DIR_HOST}:/data:ro" \
    --container-workdir="${SUBMIT_DIR}" \
    --container-remap-root --no-container-mount-home \
    bash -c '
        set -Eeuo pipefail
        export XDG_CACHE_HOME=/root/.cache
        export HF_HOME=/root/.cache/huggingface
        export HF_HUB_CACHE=/root/.cache/huggingface/hub
        export TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers
        export TRITON_CACHE_DIR=/root/.cache/triton
        export TORCH_EXTENSIONS_DIR=/root/.cache/torch-extensions
        export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor
        export CUDA_CACHE_PATH=/root/.cache/cuda
        export TMPDIR=/root/.cache/tmp
        export TORCH_NCCL_AVOID_RECORD_STREAMS=1
        export NCCL_NVLS_ENABLE=0
        mkdir -p "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$TRITON_CACHE_DIR" \
            "$TORCH_EXTENSIONS_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$TMPDIR"
        [[ -w "$RUN_OUTPUT_DIR" ]] || {
            echo "Output directory is not writable on $(hostname): $RUN_OUTPUT_DIR" >&2
            exit 2
        }

        python "${SUBMIT_DIR}/lightning35.py" --check --config "${SUBMIT_DIR}/lightning35.yaml"
        python -c "from megatron.bridge.recipes.nemotronh.nemotron_3_nano import nemotron_3_5_lightning_pretrain_config"
        GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
        [[ "$GPU_COUNT" == 8 ]] || { echo "Expected 8 visible GPUs on $(hostname), found $GPU_COUNT" >&2; exit 2; }
        echo "Node $(hostname): Slurm rank ${SLURM_PROCID}, GPUs ${GPU_COUNT}"

        exec torchrun \
            --nnodes=2 --nproc_per_node=8 --node_rank="${SLURM_PROCID}" \
            --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
            --max_restarts=0 \
            "${SUBMIT_DIR}/lightning35.py" --config "${SUBMIT_DIR}/lightning35.yaml"
    '
