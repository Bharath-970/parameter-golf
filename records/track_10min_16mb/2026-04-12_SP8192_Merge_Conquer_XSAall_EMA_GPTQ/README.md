# Merge and Conquer: SP8192 + Advanced Architectural Stack

This run targets the **1.0810 SOTA** (set by `bigbag`) by merging their tokenizer advantage with our advanced architectural stack. Our goal is a multi-seed average of **1.0760 or lower** (a 0.005 win margin).

## Strategy Analysis: SOTA vs. Our Run

| Technique | SOTA (bigbag, 1.0810) | Our Run (Merge & Conquer) |
| :--- | :--- | :--- |
| **Tokenizer** | SP8192 | **SP8192** |
| **Recurrence** | 3-Layer Recurrence | **3-Layer Depth Recurrence** (Blocks 4-6) |
| **QK-Gain** | 5.25 | **5.25** |
| **Architecture** | Parallel Residuals, TTT | Parallel Residuals, TTT, **XSA-all**, 3x MLP, Partial RoPE |
| **Training** | Standard | **High Weight Decay (0.085)**, **EMA** |
| **Quantization** | Standard (inferred) | **Self-generated GPTQ** (Activation-aware) |

## The Winning Rationale

The "Merge and Conquer" strategy combined the single biggest lever in the competition—the **SP8192 tokenizer**—with our unique architectural strengths (**XSA-all**, **MuonEq-R**, **EMA**, and **Self-Gen GPTQ**). 

By adopting the SP8192 vocabulary, we regain the estimated **~0.010 BPB** compression penalty inherent to smaller vocabularies (like SP4096), while retaining the modeling efficiency of our deep stack.

## Projected BPB Score

| Milestone | BPB Estimate |
| :--- | :--- |
| Baseline (aryanbhosale, SP8192) | 1.0822 |
| Architectural Edge (MuonEq-R, XSA-all, EMA) | -0.0020 |
| New Projection (Optimistic) | **1.0802** |
| **User "Merge" Projection** | **1.0752** |

*Target SOTA: 1.0810. Target Win: 1.0760.*

## Technical Implementation Details

- **XSA-all:** All transformer blocks share the Key and Value projections from Block 0, significantly reducing parameter count and allowing for greater depth/width within the 16MB limit.
- **EMA (0.997):** We maintain an Exponential Moving Average of weights during training, which has proven more stable for post-training quantization (PTQ) than final-step checkpoints.
- **Self-Gen GPTQ:** Instead of static calibration data, we generate representative sequences autoregressively from the trained model to find optimal quantization scales for every layer.
- **Brotli Compression:** We use Brotli-11 combined with byte-shuffling to compress the quantized model payload, ensuring the 8192-vocab embeddings fit within the 16,000,000 byte cap.

## Training Launch (8xH100)

```bash
RUN_ID=sp8192_merge_conquer_seed42 \
DATA_PATH=./data/datasets/fineweb10B_sp8192 \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \
VOCAB_SIZE=8192 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
