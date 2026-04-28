# SP8192 + SwiGLU + Warmdown Fraction + Quarter Batch

This run starts from the SP8192 Merge-and-Conquer stack and applies three proven improvements:

- **SwiGLU MLP** with hidden scaled to preserve parameter budget
- **Warmdown by wallclock fraction** (`WARMDOWN_FRAC=0.2`) to avoid early LR decay
- **Quarter batch tokens** with **optional grad accumulation** to increase optimizer steps

Goal: beat the 1.0810 SOTA with a multi-seed average of **1.0760 or lower**.

## Key Additions

- SwiGLU gated activation replaces LeakyReLU^2 in the MLP
- Warmdown based on wallclock fraction (fixes step-1 decay under 10-minute cap)
- Smaller batch tokens + accumulation to trade throughput for more steps

## Training Launch (8xH100)

```bash
RUN_ID=sp8192_swiglu_warmdown_qbatch_seed42 \
SEED=42 \
DATA_PATH=./data/datasets/fineweb10B_sp8192 \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \
VOCAB_SIZE=8192 \
TRAIN_BATCH_TOKENS=196608 \
GRAD_ACCUM_STEPS=2 \
SWIGLU_ENABLED=1 \
SWIGLU_HIDDEN_MULT=0.6667 \
WARMDOWN_FRAC=0.2 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## 3-Seed Sweep

```bash
for s in 42 1337 2024; do
	RUN_ID=sp8192_swiglu_warmdown_qbatch_seed${s} \
	SEED=$s \
	DATA_PATH=./data/datasets/fineweb10B_sp8192 \
	TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \
	VOCAB_SIZE=8192 \
	TRAIN_BATCH_TOKENS=196608 \
	GRAD_ACCUM_STEPS=2 \
	SWIGLU_ENABLED=1 \
	SWIGLU_HIDDEN_MULT=0.6667 \
	WARMDOWN_FRAC=0.2 \
	torchrun --standalone --nproc_per_node=8 train_gpt.py
done
```

## Notes

- Set `WARMDOWN_FRAC=0` to fall back to `WARMDOWN_ITERS`.
- Adjust `TRAIN_BATCH_TOKENS` and `GRAD_ACCUM_STEPS` together to control stability and step count.
