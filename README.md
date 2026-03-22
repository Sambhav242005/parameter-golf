# Depth-Recurrent Transformer with U-Net Skip Connections

## Result

| Metric | Value |
|--------|-------|
| `val_bpb` (after int8 + zlib) | **1.4061** |
| Artifact size | **4.39 MB** / 16 MB budget |
| Parameters | 4,726,284 |
| Training time | ~45 seconds (RTX 4090) |

## Approach

Two modifications from the baseline `train_gpt.py`:

### 1. Depth Recurrence

Instead of `num_layers` independent transformer blocks, a **single shared block** is applied `num_passes` times. This collapses unique parameter count from ~25M to ~4.7M while keeping the same compute depth.

```
baseline:  9 independent blocks × ~2.8M params = ~25M unique params
this:      1 shared block × 12 passes  =  ~4.7M unique params
```

Weight sharing means zlib can compress the artifact much more aggressively — the same weights are reused, not stored repeatedly.

### 2. U-Net Skip Connections

The passes are split into encoder (first half) and decoder (second half). Encoder pass outputs are stored and added back in reverse order during decoder passes, with learned per-pass scale vectors (`skip_weights`):

```python
# Encoder: store activations
for _ in range(num_encoder_passes):
    x = block(x, x0)
    skips.append(x)

# Decoder: add skip from mirror encoder pass
for i in range(num_decoder_passes):
    x = x + skip_weights[i] * skips.pop()
    x = block(x, x0)
```

This gives gradient paths that bypass the recurrence bottleneck and lets early/late passes specialize.

### 3. x0 Residual Mix

Every block call receives both `x` (current state) and `x0` (original embedding) via a learned `resid_mix` parameter. This stabilizes training across many recurrent passes and prevents representation collapse.

## Configuration Used

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
MODEL_DIM=768 NUM_HEADS=12 NUM_KV_HEADS=4 \
WARMDOWN_ITERS=600 MAX_WALLCLOCK_SECONDS=0 \
TRAIN_BATCH_TOKENS=524288 NUM_PASSES=12 \
MATRIX_LR=0.02 GRAD_CLIP_NORM=1.0 \
torchrun --standalone --nproc_per_node=1 train_gpt_recurrent.py
```

## Training Curve

```
step  1000:  val_bpb 1.5214
step  5000:  val_bpb 1.4147
step 10000:  val_bpb 1.4771  (spike recovery)
step 15000:  val_bpb 1.4620
step 18000:  val_bpb 1.4608  (best checkpoint)
step 19000:  val_bpb 1.3735  (during warmdown)
final roundtrip (int8+zlib): 1.4061
```

## Why This Works for Parameter Golf

The key insight is **compute is free, storage is not**. In the 16MB budget:

- A 9-layer independent model stores 9 × block_size weights
- A depth-recurrent model stores 1 × block_size weights, used 12 times
- zlib compresses the quantized weights further since the shared block is stored once

This frees budget to go wider (`MODEL_DIM=768` vs baseline `512`), which improves bpb more than going deeper with independent layers.

## Planned Next Steps

- Pass depth embeddings: `self.pass_emb = nn.Embedding(num_passes, model_dim)` so the shared block can specialize per depth
- AttnRes: replace fixed U-Net skips with learned attention over all prior pass outputs
- SwiGLU activation instead of squared-ReLU
- Scale to `MODEL_DIM=1024` to use more of the 16MB budget
