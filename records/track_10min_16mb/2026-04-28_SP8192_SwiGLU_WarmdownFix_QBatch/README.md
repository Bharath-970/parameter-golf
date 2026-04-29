# SP8192 + Warmdown Fraction + Optional SwiGLU (Runpod Iteration Profile)

This run starts from the SP8192 Merge-and-Conquer stack and keeps the defaults aligned
with the stronger Apr-12 profile for faster, safer iteration:

- **Proven batch size** default (`TRAIN_BATCH_TOKENS=786432`)
- **Conservative legal TTT LR** default (`TTT_SF_LR=0.0003`)
- **More frequent TTT progress logs** (`TTT_SF_PROGRESS_EVERY=100`)
- **Optional SwiGLU path** for A/B tests (`SWIGLU_ENABLED=1`)

Goal: beat the 1.0810 SOTA with a multi-seed average of **1.0760 or lower**.

## Key Defaults

- `TRAIN_BATCH_TOKENS=786432` (instead of quarter-batch)
- `TTT_SF_LR=0.0003`
- `EVAL_STRIDE=32`
- `SWIGLU_ENABLED=0` by default (opt-in)

## Run #1 (Fast Complete Score, No TTT)

```bash
RUN_ID=sp8192_seed42_nottt \
SEED=42 \
DATA_PATH=./data/datasets/fineweb10B_sp8192 \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \
VOCAB_SIZE=8192 \
TTT_SF_ENABLED=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Run #2 (Legal TTT Enabled, Better Final Score)

```bash
RUN_ID=sp8192_seed42_ttt \
SEED=42 \
DATA_PATH=./data/datasets/fineweb10B_sp8192 \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \
VOCAB_SIZE=8192 \
TTT_SF_ENABLED=1 \
TTT_SF_LR=0.0003 \
TTT_SF_STEPS=3 \
TTT_SF_PROGRESS_EVERY=100 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Notes

- Use `SWIGLU_ENABLED=1` only for A/B tests after a stable no-SwiGLU baseline.
- If post-train TTT appears idle, check logs for `ttt_sf [...]` progress lines every 100 docs.
