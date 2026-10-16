# Gemma 4 E4B: MFU benchmark on 8× B300 (one node)

These steps pre-train **Gemma 4 E4B** (text path) on dummy (mock) data with
[Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/gemma/gemma4)
on one node with 8 B300 GPUs. A small script then turns the training log into
**Model FLOPs Utilization (MFU)**.

The run uses the upstream recipe `gemma4_e4b_pretrain_config` inside the
`nvcr.io/nvidia/nemo:26.08` container, on Slurm partition `req-635` (node `exp-blr-dgxb300-01`).
Nothing in `Megatron-Bridge/` is modified. The speed-ups come from Slurm settings, recipe
overrides, and runtime patches applied by `scripts/run_gemma4_fast.py`.

**Result (measured):** with the step 5 configuration, 8× B300 reach **≈ 30% MFU**
(1.41 s/step, 23,266 tokens/s/GPU, about 674 model TFLOP/s per GPU). The unmodified recipe
reached 3.9%. See [Performance changes](#performance-changes).

```
gemma/
├── README.md
├── scripts/run_gemma4_fast.py   <- launcher: runtime speed patches, then runs run_recipe.py
├── scripts/compute_mfu.py       <- turns the training log into MFU
├── Megatron-Bridge/             <- created in step 1 (unmodified)
└── runs/                        <- created in step 5 (train.log, mfu_report.txt)
```

## Prerequisites

- Access to Slurm partition and account `req-635`.
- A Hugging Face token at `~/.cache/huggingface/token` (create it with `hf auth login`).
  You must also accept the license at <https://huggingface.co/google/gemma-4-E4B-it>.
  Only the model's `config.json` is downloaded, not the weights.

## Step 1: Get the code (login shell)

```bash
cd /userhome/home/akumarkuruba/gemma
git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git      # skip if Megatron-Bridge/ exists
git -C Megatron-Bridge submodule update --init --depth 1 3rdparty/Megatron-LM
```

Gemma 4 requires the Megatron-LM checkout that Bridge pins as a submodule on `PYTHONPATH`.

## Step 2: Free the node

The training needs all 8 GPUs, so none of your other jobs may hold a GPU on the node:

```bash
squeue -p req-635
scontrol show node exp-blr-dgxb300-01 | grep AllocTRES     # expect gres/gpu=0
```

If one of your jobs is listed, exit it or run `scancel <jobid>`.

## Step 3: Open a shell in the 26.08 container on 8 GPUs and all CPUs

```bash
srun -p req-635 -A req-635 -N 1 --ntasks=1 --cpus-per-task=256 --gpus-per-node=8 --exclusive \
     --time=02:00:00 \
     --container-image=nvcr.io#nvidia/nemo:26.08 \
     --container-name=gemma4-nemo2608 \
     --container-mounts=/userhome:/userhome,/raid:/raid \
     --container-workdir=/userhome/home/akumarkuruba/gemma \
     --pty bash --norc
```

- **`--cpus-per-task=256` is essential.** Without it, Slurm pins the shell, and every
  process started from it, to 2 CPU threads. All 8 training processes then share those 2
  threads, and training runs 3–4× slower. To check, run
  `grep Cpus_allowed_list /proc/self/status` in the shell; it should print `0-255`.
- `--norc` skips your `~/.bashrc`, which runs `module load ...` and fails inside the container
  (`/usr/bin/tclsh: No such file or directory`).
- The first time, Pyxis pulls the image, which takes several minutes.
- `--container-name` reuses the same container the next time you run this command.
- All remaining steps run **inside this shell**.

## Step 4: Prepare the environment (inside the container)

```bash
nvidia-smi -L                                         # expect 8 x NVIDIA B300

# Gemma 4 needs transformers >= 5.15 (the image has 5.12). The image's /opt/venv is
# read-only, so install the newer packages into a folder on /raid (first time only).
python -m pip install --no-deps --target /raid/akumarkuruba/pylibs \
    "transformers>=5.15,<5.18" "tokenizers>=0.23.1,<0.24"

export GEMMA_ROOT=/userhome/home/akumarkuruba/gemma
export BRIDGE_ROOT=$GEMMA_ROOT/Megatron-Bridge
export PYTHONPATH=/raid/akumarkuruba/pylibs:$BRIDGE_ROOT/src:$BRIDGE_ROOT/3rdparty/Megatron-LM:${PYTHONPATH:-}
export HF_HOME=/raid/akumarkuruba/hf_cache
export HF_TOKEN=$(cat /userhome/home/akumarkuruba/.cache/huggingface/token)
export CUDA_DEVICE_MAX_CONNECTIONS=1

python -c "import torch, transformers, megatron.bridge; print(torch.__version__, transformers.__version__, torch.cuda.device_count())"
grep Cpus_allowed_list /proc/self/status               # expect 0-255
```

The `python` command should print a transformers version of 5.15 or later, followed by `8`.

## Step 5: Train on dummy data (8 GPUs)

```bash
export TP=1 SEQ_LEN=4096 MBS=2 GBS=64 TRAIN_ITERS=100
export RUN_DIR=$GEMMA_ROOT/runs/tp${TP}_mbs${MBS}_gbs${GBS}_seq${SEQ_LEN}
mkdir -p $RUN_DIR && cd $RUN_DIR

python -m torch.distributed.run --nproc_per_node 8 \
    $GEMMA_ROOT/scripts/run_gemma4_fast.py \
    --recipe gemma4_e4b_pretrain_config --mode pretrain --dataset mock \
    -tp $TP -sl $SEQ_LEN -mb $MBS -gb $GBS -ms $TRAIN_ITERS --warmup_iters 10 \
    checkpoint.save=null checkpoint.load=null checkpoint.pretrained_checkpoint=null \
    logger.log_interval=1 logger.log_throughput=true validation.eval_iters=0 \
    ddp.overlap_grad_reduce=false ddp.overlap_param_gather=false \
    ddp.grad_reduce_in_fp32=false ddp.check_for_nan_in_grad=false \
    model.gradient_accumulation_fusion=true model.apply_rope_fusion=true \
    2>&1 | tee $RUN_DIR/train.log
```

The first two iterations are slow (about 1–2 minutes) while `torch.compile` builds its kernels.
After that, each step takes about 1.4 s.

`run_gemma4_fast.py` takes the same arguments as Bridge's `scripts/training/run_recipe.py`.
It patches a few functions in memory, prints `[gemma4-fast] ...` lines for them, and then
runs `run_recipe.py` unchanged. To run the plain upstream code instead, replace
`$GEMMA_ROOT/scripts/run_gemma4_fast.py` with
`$BRIDGE_ROOT/scripts/training/run_recipe.py`.

| Variable | Meaning |
|---|---|
| `TP` | Tensor parallel size. With TP=1 all 8 GPUs are data parallel (fastest for this model). |
| `SEQ_LEN` | Sequence length |
| `MBS` | Micro batch size per GPU. 2 is the fastest that fits; 4 runs out of memory. |
| `GBS` | Global batch size. Must be a multiple of `MBS × 8 / TP`. |
| `TRAIN_ITERS` | Number of training steps |

A healthy run prints one line per iteration:

```
iteration       42/     100 | ... | elapsed time per iteration (ms): 1408.4 | ... | lm loss: ...
```

## Step 6: Compute MFU

```bash
python3 $GEMMA_ROOT/scripts/compute_mfu.py --log $RUN_DIR/train.log \
    --seq-len $SEQ_LEN --gbs $GBS --num-gpus 8 --peak-tflops 2250 --skip-iters 20 \
    | tee $RUN_DIR/mfu_report.txt
```

The `MFU` line is the result. Type `exit` to leave the container and release the GPUs.

To stop a running training: press Ctrl+C in the container shell. Don't use Ctrl+Z; that only
pauses training, and it keeps holding all GPU memory. To end the whole Slurm job from any
shell, run `squeue -u $USER`, then `scancel <jobid>`.

## Performance changes

All results below are measured on this node: 8× B300, GBS=64, sequence 4096, median step time
after warm-up. Nothing in `Megatron-Bridge/` was changed.

| # | Change | Where | Step time | MFU |
|---|---|---|---|---|
| 0 | Upstream recipe as-is (MBS=1, 2 CPU threads) | – | 10.95 s | 3.9% |
| 1 | Give the job all CPUs | `srun --cpus-per-task=256` (step 3) | 2.92 s | 14.4% |
| 2 | Fused RMSNorm, fused logit softcap, flex attention for sliding layers, MBS=2 | `run_gemma4_fast.py`, `MBS=2` | 1.54 s | 27.4% |
| 3 | Gradient-accumulation fusion, BF16 gradient reduce, no NaN check | step 5 overrides | 1.50 s | 28.1% |
| 4 | Fused RoPE | `model.apply_rope_fusion=true` | **1.41 s** | **≈ 30%** |

What each change does:

1. **CPU pinning (largest gain).** `srun --ntasks=1` without `--cpus-per-task` gives the step
   one core (2 hardware threads). All 8 training processes, their data-loader workers, and the
   CUDA kernel-launch threads competed for those 2 threads, so the GPUs sat idle about 70% of
   the time (the 30% utilization in `nvidia-smi`).
2. **Runtime patches in `run_gemma4_fast.py`.** Each patch can be turned off by setting its
   variable to `0`:
   - `GEMMA4_FAST_NORM`: E4B has 8 RMSNorms per layer (4 layer norms, Q/K/V norms, PLE norm).
     Bridge computes each in FP32 with about 6 separate kernels. `torch.compile` turns each
     into one fused kernel with the same math.
   - `GEMMA4_FAST_SOFTCAP`: the final `30·tanh(logits/30)` runs over a
     4096 × 262,144-vocab logit tensor; compiled into one kernel.
   - `GEMMA4_FAST_ATTN`: the 35 sliding-window layers only look at 512 keys, but the plain
     attention computes and masks the full 4096 × 4096 matrix. `flex_attention` with a
     block-sparse window skips the masked blocks. Attention for these layers became 3×
     faster, and the output matches the original to bf16 precision. The 7 global layers
     (head_dim 512) keep the original attention: no fast kernel in this container supports
     head_dim 512.
   - **MBS=2:** fewer, larger kernels per token. This cuts the CPU launch overhead and
     raises GEMM efficiency.
3. **Overrides.** `model.gradient_accumulation_fusion=true` adds weight gradients into the
   FP32 buffer inside the GEMM, which removes about 1,500 extra add kernels per step.
   `ddp.grad_reduce_in_fp32=false` halves gradient traffic, and
   `ddp.check_for_nan_in_grad=false` skips a per-step check. These last two are small gains.
4. **`model.apply_rope_fusion=true`.** Uses the fused RoPE kernel instead of
   slice/rotate/`torch.cat`. Gemma 4's partial RoPE on the global layers stores zero
   frequencies for the non-rotated dimensions, so the fused kernel gives the same result.
   Losses matched the unfused run to within run-to-run noise.

**Tried and not kept:**

- `ddp.overlap_grad_reduce=true`: 3% slower here, because NVLink makes the gradient sync
  cheap. It also crashes unless `GEMMA4_FREEZE_UNUSED=1`. See the troubleshooting table.
- `MBS=4`: out of memory.
- Flash, cuDNN and memory-efficient SDPA: PyTorch reports "No available kernel" for these
  shapes.
- flex attention for the head_dim-512 global layers: crashes with a CUDA misaligned-address
  error.

**What limits MFU now** (profile of the step 5 configuration):

- GEMMs: 39% of GPU time.
- Element-wise ops: 24%.
- Flex attention: 20%.
- The 7 global layers' unfused attention (softmax and mask): about 6%.

Going well beyond 30% needs a Transformer Engine version of the Gemma 4 E4B layer (fused
attention, fused norm + linear, FP8, CUDA graphs). That means a change inside Megatron-Bridge.

## How MFU is computed

```
MFU = training FLOPs per step / (step time × 8 GPUs × peak BF16 FLOP/s per GPU)
```

- **Training FLOPs** equal 3 × forward FLOPs (no activation recompute).
- **Forward FLOPs** are counted from the E4B architecture. The count covers:
  - Q/K/V/O projections (K/V skipped on the 18 KV-shared layers),
  - causal attention (512-token sliding window on 35 layers, 512-dim heads on the 7 global layers),
  - the gated GeLU MLP,
  - the Per-Layer Embeddings,
  - the 262,144-vocab LM head.

  At seq 4096 this gives about **29.0 GFLOP per training token**.
- **Step time** is the median `elapsed time per iteration` after the first 20 iterations.
- **Peak** is `--peak-tflops 2250`, which is the dense BF16 figure assumed for B300. Check the
  datasheet for this SKU (`B300 SXM6 AC`) and adjust if needed.
- Megatron-Bridge also logs its own `throughput per GPU (TFLOP/s/GPU)`, shown for reference.
  It uses a generic GPT formula that doesn't model Gemma 4's architecture.

## CUDA and optimization flags

This section lists the settings in effect with the step 5 command: recipe
`gemma4_e4b_pretrain_config`, plus the step 5 overrides, plus the `run_gemma4_fast.py`
patches, at TP=1. To confirm them for a run, check the config dump at the top of
`train.log`.

> **Important:** the Gemma 4 E4B layer definition (`get_gemma4_layer_spec` in
> `Megatron-Bridge/src/megatron/bridge/models/gemma/modeling_gemma4.py`) always builds the
> plain PyTorch ("local") version of each layer, not the Transformer Engine (TE) version.
> Features marked "Needs TE" can't be switched on with a flag today; they need a code change
> in Megatron-Bridge.

### Enabled

| Flag | Enabled | What it does |
|---|---|---|
| `srun --cpus-per-task=256` | Yes | Gives the training processes all CPU threads instead of 2. The largest single speed-up. |
| `mixed_precision=bf16_mixed` | Yes | Computes in BF16, which roughly doubles speed over FP32. Optimizer state stays in FP32. |
| `ddp.use_distributed_optimizer` | Yes | Splits the optimizer state (Adam moments, FP32 weights) across the 8 GPUs to save memory. |
| `ddp.average_in_collective` | Yes | Averages gradients inside the communication call itself instead of in a separate step. |
| `model.gradient_accumulation_fusion` | Yes | Adds weight gradients into the FP32 buffer inside the matrix-multiply kernel, saving a memory pass. |
| `model.apply_rope_fusion` | Yes | Applies rotary position embeddings with one fused kernel. |
| `cross_entropy_loss_fusion` (`native`) | Yes | Computes the loss over the 262k-word vocabulary in one fused step, saving memory and time. |
| Compiled RMSNorm (`GEMMA4_FAST_NORM`) | Yes | The 8 RMSNorms per layer each run as one fused kernel (same math), via `torch.compile`. |
| Compiled logit softcap (`GEMMA4_FAST_SOFTCAP`) | Yes | `30·tanh(x/30)` on the logits runs as one fused kernel. |
| Flex attention for sliding layers (`GEMMA4_FAST_ATTN`) | Yes | The 35 sliding-window layers compute only the 512-key window, using a fused block-sparse kernel. |
| `train.manual_gc` (every 100 steps) | Yes | Runs Python garbage collection on a fixed schedule so it doesn't cause random stalls. |
| `CUDA_DEVICE_MAX_CONNECTIONS=1` | Yes | Makes the GPU run work in the order it was queued, so compute and communication overlap predictably. |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Yes | Lets PyTorch grow memory blocks instead of fragmenting them, which reduces out-of-memory errors. |
| `TORCH_NCCL_AVOID_RECORD_STREAMS=1` | Yes | Frees communication buffers sooner, lowering memory use. |
| `TORCH_NCCL_HIGH_PRIORITY=1` | Yes | Runs GPU-to-GPU communication on high-priority streams so it isn't delayed by compute. |

### Not enabled

| Flag | Enabled | What it does |
|---|---|---|
| `ddp.grad_reduce_in_fp32` | No | Would send gradients between GPUs in FP32 (twice the traffic). Step 5 turns it off and uses BF16. |
| `ddp.check_for_nan_in_grad` | No | Would check gradients for NaN on every step. Step 5 turns it off for speed; turn it back on for real training. |
| `ddp.overlap_grad_reduce` | No | Sends gradients between GPUs during the backward pass. Measured 3% slower on this node. Also crashes unless `GEMMA4_FREEZE_UNUSED=1` (see Troubleshooting). |
| `ddp.overlap_param_gather` | No | Collects updated weights from other GPUs while the next forward pass starts. Off together with `overlap_grad_reduce`. |
| Fused attention for the 7 global layers | No | The global layers use head_dim 512, which no fast attention kernel in this container supports. They keep the plain attention, which stores a full 4096×4096 matrix per head. |
| `transformer_impl=transformer_engine` | No | Uses NVIDIA's TE-optimized layers (fused norm + linear, faster matrix multiplies). Not available for E4B today. Needs TE. |
| FP8 / MXFP8 (`mixed_precision=bf16_with_fp8_...`) | No | Matrix multiplies in 8-bit, about 2× BF16 throughput on B300. Needs TE. |
| CUDA graphs (`cuda_graph_impl`) | No | Records a whole step once and replays it, cutting CPU launch overhead. Needs TE. |
| `masked_softmax_fusion` | No | Fuses the attention mask and softmax into one kernel. The recipe turns it off for the plain layers. |
| `bias_activation_fusion` | No | Fuses the GeLU activation with the MLP's gating multiply. Megatron's fused kernel doesn't cover Gemma's tanh-GeLU. |
| `persist_layer_norm` | No | Apex LayerNorm kernel. Not used by Gemma 4's own RMSNorm class; the compiled RMSNorm covers it. |
| `bias_dropout_fusion` | No | Fuses bias + dropout + residual add. Little effect here: E4B has no bias and dropout is 0. |
| `attention_softmax_in_fp32` | No | Runs softmax in FP32 for accuracy. Off is faster. |
| `NCCL_NVLS_ENABLE` (set to 0 by the recipe) | No | NVLink SHARP: does part of the gradient all-reduce inside the NVSwitch. |
| `NCCL_GRAPH_REGISTER` (set to 0 by the recipe) | No | Pre-registers communication buffers with NCCL for faster transfers. |
| `optimizer.use_precision_aware_optimizer` | No | Keeps optimizer state in lower precision to save memory. |
| Activation recompute (`recompute_granularity`) | No | Recomputes activations in the backward pass instead of storing them: saves memory, costs about 30% extra compute. Off is right while memory allows. |
| Activation / CPU offloading | No | Moves activations to CPU memory to free GPU memory. Not needed with 288 GB GPUs. |
| `model.sequence_parallel` | No | Splits norm and dropout work across tensor-parallel GPUs. Only matters with TP>1. |
| `tp_comm_overlap` | No | Overlaps tensor-parallel communication with matrix multiplies. Only matters with TP>1; needs TE. |
| Megatron FSDP (`use_megatron_fsdp`) | No | Fully shards weights, gradients and optimizer state across GPUs. Not needed for a 4B model. |

## Troubleshooting

| Symptom | Fix |
|---|---|
| Very slow steps (about 11 s) and low GPU utilization in `nvidia-smi` | The shell has only 2 CPU threads. Restart step 3 with `--cpus-per-task=256`, then check `grep Cpus_allowed_list /proc/self/status` (expect `0-255`). |
| `bash: /usr/bin/tclsh: No such file or directory` | Harmless; comes from `module load` in `~/.bashrc`. Use `--pty bash --norc` (step 3). |
| `401` / `GatedRepoError` for `google/gemma-4-E4B-it` | Accept the license on Hugging Face, then run `hf auth login`. |
| `transformers` doesn't recognize `gemma4_text`, or the version check prints 5.12 | Rerun the `pip install` from step 4 and the `export PYTHONPATH=/raid/akumarkuruba/pylibs:...` line. |
| `srun` waits in `PENDING` | Another job holds GPUs on the node; see step 2. |
| `global batch size ... not divisible` | Make `GBS` a multiple of `MBS × 8 / TP`. |
| CUDA out of memory | Use `MBS=1` (and keep `GBS` a multiple of 8). |
| `AssertionError` at `assert len(self.per_param_grad_ready_counts) == len(self.params)` after iteration 1 | Happens only with `ddp.overlap_grad_reduce=true`. The `k_layernorm` weights in the 18 KV-shared layers never get a gradient, because their K is discarded. Keep overlap off (as in step 5), or set `GEMMA4_FREEZE_UNUSED=1` before the step 5 command to freeze those weights. |
| A `[gemma4-fast]` patch fails or you suspect it | Turn it off with `GEMMA4_FAST_NORM=0`, `GEMMA4_FAST_SOFTCAP=0` or `GEMMA4_FAST_ATTN=0` before the step 5 command. |
| `no 'iteration ...' lines found` | Training failed before it started logging; check `train.log` for the error. |
