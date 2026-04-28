# SP8192 + Hessian-Mix + Brotli/SWA Attack Run

This run targets beating `1.0810` with margin by stacking orthogonal gains:

- `SP8192` tokenizer/data
- 3-layer recurrence (`RECUR_START=4`, `RECUR_END=7`)
- parallel residuals + legal score-first TTT
- SWA late in training
- Hessian-proxy ranked mixed `Int5/Int6` GPTQ (Int5 on least-sensitive MLP layers)
- Brotli + byte-shuffle model payload compression
- coprime-stride shard traversal

## Default launch

```bash
RUN_ID=sp8192_hessmix_brotli_swa_seed42 \
SEED=42 \
DATA_PATH=./data/datasets/fineweb10B_sp8192 \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \
VOCAB_SIZE=8192 \
COMPRESSOR=brotli \
BYTE_SHUFFLE=1 \
SWA_ENABLED=1 \
HESSIAN_RANK_ENABLED=1 \
INT5_MLP_FRAC=0.5 \
COPRIME_LOADER_ENABLED=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## 3-seed significance sweep

```bash
for s in 42 1337 2024; do
  RUN_ID=sp8192_hessmix_seed${s} \
  SEED=$s \
  DATA_PATH=./data/datasets/fineweb10B_sp8192 \
  TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \
  VOCAB_SIZE=8192 \
  COMPRESSOR=brotli BYTE_SHUFFLE=1 \
  SWA_ENABLED=1 HESSIAN_RANK_ENABLED=1 INT5_MLP_FRAC=0.5 \
  COPRIME_LOADER_ENABLED=1 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
done
```

## Margin goal

Target submission mean should be at least `<= 1.0760` (0.005 better than `1.0810`) with reproducibility across seeds.
