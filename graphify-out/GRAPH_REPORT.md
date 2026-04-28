# Graph Report - .  (2026-04-28)

## Corpus Check
- 71 files · ~210,019 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1886 nodes · 3767 edges · 31 communities detected
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 360 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_12L XSA4 GPTQ|12L XSA4 GPTQ]]
- [[_COMMUNITY_11L XSAall GPTQ|11L XSAall GPTQ]]
- [[_COMMUNITY_14L XSA6 TTT|14L XSA6 TTT]]
- [[_COMMUNITY_16L XSAall TTT|16L XSAall TTT]]
- [[_COMMUNITY_14L XSA6 GPTQ|14L XSA6 GPTQ]]
- [[_COMMUNITY_SP8192 HessMix SWA|SP8192 HessMix SWA]]
- [[_COMMUNITY_13L XSA4 GPTQ|13L XSA4 GPTQ]]
- [[_COMMUNITY_12L EMA Trigram|12L EMA Trigram]]
- [[_COMMUNITY_12L Int4 Trigram|12L Int4 Trigram]]
- [[_COMMUNITY_LoRA TTT|LoRA TTT]]
- [[_COMMUNITY_11L Int4 QAT|11L Int4 QAT]]
- [[_COMMUNITY_SP4096 DepthRecur|SP4096 DepthRecur]]
- [[_COMMUNITY_SP8192 Merge-Conquer|SP8192 Merge-Conquer]]
- [[_COMMUNITY_Int6 SmearGate MLP|Int6 SmearGate MLP]]
- [[_COMMUNITY_MixedQuant SlidingWindow|MixedQuant SlidingWindow]]
- [[_COMMUNITY_SmearGate OrthoInit|SmearGate OrthoInit]]
- [[_COMMUNITY_10L Int5 MLP|10L Int5 MLP]]
- [[_COMMUNITY_Int6 QAT SlidingWindow|Int6 QAT SlidingWindow]]
- [[_COMMUNITY_SlidingWindow Eval|SlidingWindow Eval]]
- [[_COMMUNITY_Core training scripts|Core training scripts]]
- [[_COMMUNITY_Seq2048 FP16 Emb|Seq2048 FP16 Emb]]
- [[_COMMUNITY_SlidingWindow FP16 Emb|SlidingWindow FP16 Emb]]
- [[_COMMUNITY_Warmdown Quantization|Warmdown Quantization]]
- [[_COMMUNITY_Core model modules|Core model modules]]
- [[_COMMUNITY_Lower LR|Lower LR]]
- [[_COMMUNITY_FP16 Embed WD|FP16 Embed WD]]
- [[_COMMUNITY_LongContext Seq2048|LongContext Seq2048]]
- [[_COMMUNITY_Training Opt Seq4096|Training Opt Seq4096]]
- [[_COMMUNITY_10L Mixed Precision|10L Mixed Precision]]
- [[_COMMUNITY_Naive Baseline|Naive Baseline]]
- [[_COMMUNITY_Quasi10B SP1024|Quasi10B SP1024]]

## God Nodes (most connected - your core abstractions)
1. `rms_norm()` - 115 edges
2. `train()` - 93 edges
3. `get()` - 87 edges
4. `compress()` - 34 edges
5. `main()` - 24 edges
6. `main()` - 21 edges
7. `main()` - 20 edges
8. `main()` - 20 edges
9. `main()` - 20 edges
10. `main()` - 19 edges

## Surprising Connections (you probably didn't know these)
- `test_mixed_quantize_roundtrip()` --calls--> `get()`  [INFERRED]
  smoke_test.py → data/cached_challenge_fineweb.py
- `test_model_forward()` --calls--> `GPT`  [INFERRED]
  smoke_test.py → records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/train_gpt.py
- `test_model_forward()` --calls--> `train()`  [INFERRED]
  smoke_test.py → run_modal.py
- `test_artifact_size()` --calls--> `GPT`  [INFERRED]
  smoke_test.py → records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/train_gpt.py
- `build_model()` --calls--> `GPT`  [INFERRED]
  check_budget.py → records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/train_gpt.py

## Communities

### Community 0 - "12L XSA4 GPTQ"
Cohesion: 0.04
Nodes (57): Hyperparameters, apply_rotary_emb(), BigramHashEmbedding, Block, build_sentencepiece_luts(), CastedLinear, CausalSelfAttention, _classify_param() (+49 more)

### Community 1 - "11L XSAall GPTQ"
Cohesion: 0.04
Nodes (59): apply_rotary_emb(), BigramHashEmbedding, Block, build_sentencepiece_luts(), CastedLinear, CausalSelfAttention, _classify_param(), dequantize_mixed_int6() (+51 more)

### Community 2 - "14L XSA6 TTT"
Cohesion: 0.04
Nodes (50): apply_rotary_emb(), BatchedLinearLoRA, BatchedTTTLoRA, BigramHashEmbedding, Block, build_sentencepiece_luts(), _build_ttt_optimizer(), CastedLinear (+42 more)

### Community 3 - "16L XSAall TTT"
Cohesion: 0.05
Nodes (42): apply_rotary_emb(), BigramHashEmbedding, Block, build_sentencepiece_luts(), CastedLinear, CausalSelfAttention, _classify_param(), dequantize_mixed_int6() (+34 more)

### Community 4 - "14L XSA6 GPTQ"
Cohesion: 0.05
Nodes (40): apply_rotary_emb(), BigramHashEmbedding, Block, build_sentencepiece_luts(), CastedLinear, CausalSelfAttention, _classify_param(), dequantize_mixed_int6() (+32 more)

### Community 5 - "SP8192 HessMix SWA"
Cohesion: 0.05
Nodes (48): apply_rotary_emb(), Block, build_sentencepiece_luts(), _byte_shuffle(), _byte_unshuffle(), CastedLinear, CausalSelfAttention, _classify_param() (+40 more)

### Community 6 - "13L XSA4 GPTQ"
Cohesion: 0.05
Nodes (40): apply_rotary_emb(), BigramHashEmbedding, Block, build_sentencepiece_luts(), CastedLinear, CausalSelfAttention, _classify_param(), dequantize_mixed_int6() (+32 more)

### Community 7 - "12L EMA Trigram"
Cohesion: 0.05
Nodes (39): apply_rotary_emb(), BigramHashEmbedding, Block, build_sentencepiece_luts(), CastedLinear, CausalSelfAttention, _classify_param(), dequantize_mixed_int6() (+31 more)

### Community 8 - "12L Int4 Trigram"
Cohesion: 0.05
Nodes (39): apply_rotary_emb(), BigramHashEmbedding, Block, build_sentencepiece_luts(), CastedLinear, CausalSelfAttention, _classify_param(), dequantize_mixed_int6() (+31 more)

### Community 9 - "LoRA TTT"
Cohesion: 0.06
Nodes (40): _accumulate_bpb(), apply_rotary_emb(), BatchedLinearLoRA, BatchedTTTLoRA, Block, build_sentencepiece_luts(), _build_ttt_optimizer(), CastedLinear (+32 more)

### Community 10 - "11L Int4 QAT"
Cohesion: 0.06
Nodes (37): apply_rotary_emb(), BigramHashEmbedding, Block, build_sentencepiece_luts(), CastedLinear, CausalSelfAttention, _classify_param(), dequantize_mixed_int6() (+29 more)

### Community 11 - "SP4096 DepthRecur"
Cohesion: 0.06
Nodes (40): apply_rotary_emb(), Block, build_sentencepiece_luts(), CastedLinear, CausalSelfAttention, _classify_param(), dequantize_mixed_int6(), DistributedTokenLoader (+32 more)

### Community 12 - "SP8192 Merge-Conquer"
Cohesion: 0.06
Nodes (40): apply_rotary_emb(), Block, build_sentencepiece_luts(), CastedLinear, CausalSelfAttention, _classify_param(), dequantize_mixed_int6(), DistributedTokenLoader (+32 more)

### Community 13 - "Int6 SmearGate MLP"
Cohesion: 0.06
Nodes (32): apply_rotary_emb(), BigramHashEmbedding, Block, build_sentencepiece_luts(), CastedLinear, CausalSelfAttention, _classify_param(), dequantize_mixed_int6() (+24 more)

### Community 14 - "MixedQuant SlidingWindow"
Cohesion: 0.06
Nodes (37): apply_rotary_emb(), Block, build_sentencepiece_luts(), CastedLinear, CausalSelfAttention, dequantize_state_dict_int8(), DistributedTokenLoader, eval_val() (+29 more)

### Community 15 - "SmearGate OrthoInit"
Cohesion: 0.06
Nodes (33): apply_rotary_emb(), BigramHash, Block, build_sentencepiece_luts(), CastedLinear, CausalSelfAttention, dequantize_state_dict_int8(), DistributedTokenLoader (+25 more)

### Community 16 - "10L Int5 MLP"
Cohesion: 0.06
Nodes (31): apply_rotary_emb(), BigramHashEmbedding, Block, build_sentencepiece_luts(), CastedLinear, CausalSelfAttention, _classify_param(), dequantize_mixed_int6() (+23 more)

### Community 17 - "Int6 QAT SlidingWindow"
Cohesion: 0.06
Nodes (36): apply_rotary_emb(), Block, BlockLARS, build_sentencepiece_luts(), CastedLinear, CausalSelfAttention, dequantize_state_dict_int8(), DistributedTokenLoader (+28 more)

### Community 18 - "SlidingWindow Eval"
Cohesion: 0.06
Nodes (33): apply_rotary_emb(), AttentionLoRA, Block, build_sentencepiece_luts(), CastedLinear, CausalSelfAttention, dequantize_state_dict_int8(), DistributedTokenLoader (+25 more)

### Community 19 - "Core training scripts"
Cohesion: 0.07
Nodes (27): accumulate_flat_grads(), Block, build_sentencepiece_luts(), CastedLinear, CausalSelfAttention, clip_grad_tree(), dequantize_state_dict_int8(), eval_val() (+19 more)

### Community 20 - "Seq2048 FP16 Emb"
Cohesion: 0.07
Nodes (31): apply_rotary_emb(), Block, build_sentencepiece_luts(), CastedLinear, CausalSelfAttention, dequantize_state_dict_int8(), DistributedTokenLoader, eval_val() (+23 more)

### Community 21 - "SlidingWindow FP16 Emb"
Cohesion: 0.07
Nodes (29): apply_rotary_emb(), Block, build_sentencepiece_luts(), CastedLinear, CausalSelfAttention, dequantize_state_dict_int8(), DistributedTokenLoader, eval_val() (+21 more)

### Community 22 - "Warmdown Quantization"
Cohesion: 0.08
Nodes (28): apply_rotary_emb(), Block, build_sentencepiece_luts(), CastedLinear, CausalSelfAttention, dequantize_state_dict_int8(), DistributedTokenLoader, eval_val() (+20 more)

### Community 23 - "Core model modules"
Cohesion: 0.08
Nodes (26): apply_rotary_emb(), Block, build_sentencepiece_luts(), CastedLinear, CausalSelfAttention, dequantize_state_dict_int8(), DistributedTokenLoader, eval_val() (+18 more)

### Community 24 - "Lower LR"
Cohesion: 0.08
Nodes (26): apply_rotary_emb(), Block, build_sentencepiece_luts(), CastedLinear, CausalSelfAttention, dequantize_state_dict_int8(), DistributedTokenLoader, eval_val() (+18 more)

### Community 25 - "FP16 Embed WD"
Cohesion: 0.08
Nodes (26): apply_rotary_emb(), Block, build_sentencepiece_luts(), CastedLinear, CausalSelfAttention, dequantize_state_dict_int8(), DistributedTokenLoader, eval_val() (+18 more)

### Community 26 - "LongContext Seq2048"
Cohesion: 0.08
Nodes (26): apply_rotary_emb(), Block, build_sentencepiece_luts(), CastedLinear, CausalSelfAttention, dequantize_state_dict_int8(), DistributedTokenLoader, eval_val() (+18 more)

### Community 27 - "Training Opt Seq4096"
Cohesion: 0.08
Nodes (26): apply_rotary_emb(), Block, build_sentencepiece_luts(), CastedLinear, CausalSelfAttention, dequantize_state_dict_int8(), DistributedTokenLoader, eval_val() (+18 more)

### Community 28 - "10L Mixed Precision"
Cohesion: 0.08
Nodes (26): apply_rotary_emb(), Block, build_sentencepiece_luts(), CastedLinear, CausalSelfAttention, dequantize_state_dict_int8(), DistributedTokenLoader, eval_val() (+18 more)

### Community 29 - "Naive Baseline"
Cohesion: 0.08
Nodes (26): apply_rotary_emb(), Block, build_sentencepiece_luts(), CastedLinear, CausalSelfAttention, dequantize_state_dict_int8(), DistributedTokenLoader, eval_val() (+18 more)

### Community 30 - "Quasi10B SP1024"
Cohesion: 0.08
Nodes (25): apply_rotary_emb(), Block, build_sentencepiece_luts(), CastedLinear, CausalSelfAttention, dequantize_state_dict_int8(), DistributedTokenLoader, eval_val() (+17 more)

## Knowledge Gaps
- **160 isolated node(s):** `Local smoke test for the Int4 QAT submission.  Tests without CUDA (works on CPU`, `Import the submission as a module, skipping the CUDA main().`, `Exact artifact size check for the Int4MLP 11L QAT submission.  Builds a full-siz`, `CPU-only check of what's cached in the volume. Costs ~$0.02.`, `Download dataset/tokenizer into the volume once. CPU-only, no GPU cost.     Poin` (+155 more)
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `get()` connect `MixedQuant SlidingWindow` to `12L XSA4 GPTQ`, `11L XSAall GPTQ`, `14L XSA6 TTT`, `16L XSAall TTT`, `14L XSA6 GPTQ`, `SP8192 HessMix SWA`, `13L XSA4 GPTQ`, `12L EMA Trigram`, `12L Int4 Trigram`, `LoRA TTT`, `11L Int4 QAT`, `SP4096 DepthRecur`, `SP8192 Merge-Conquer`, `Int6 SmearGate MLP`, `SmearGate OrthoInit`, `10L Int5 MLP`, `Int6 QAT SlidingWindow`, `SlidingWindow Eval`, `Core training scripts`, `Seq2048 FP16 Emb`, `SlidingWindow FP16 Emb`, `Warmdown Quantization`, `Core model modules`, `Lower LR`, `FP16 Embed WD`, `LongContext Seq2048`, `Training Opt Seq4096`, `10L Mixed Precision`, `Naive Baseline`, `Quasi10B SP1024`?**
  _High betweenness centrality (0.307) - this node is a cross-community bridge._
- **Why does `train()` connect `11L XSAall GPTQ` to `12L XSA4 GPTQ`, `14L XSA6 TTT`, `16L XSAall TTT`, `14L XSA6 GPTQ`, `SP8192 HessMix SWA`, `13L XSA4 GPTQ`, `12L EMA Trigram`, `12L Int4 Trigram`, `LoRA TTT`, `11L Int4 QAT`, `SP4096 DepthRecur`, `SP8192 Merge-Conquer`, `Int6 SmearGate MLP`, `MixedQuant SlidingWindow`, `SmearGate OrthoInit`, `10L Int5 MLP`, `Int6 QAT SlidingWindow`, `SlidingWindow Eval`, `Seq2048 FP16 Emb`, `SlidingWindow FP16 Emb`, `Warmdown Quantization`, `Core model modules`, `Lower LR`, `FP16 Embed WD`, `LongContext Seq2048`, `Training Opt Seq4096`, `10L Mixed Precision`, `Naive Baseline`, `Quasi10B SP1024`?**
  _High betweenness centrality (0.286) - this node is a cross-community bridge._
- **Why does `rms_norm()` connect `Int6 SmearGate MLP` to `12L XSA4 GPTQ`, `11L XSAall GPTQ`, `14L XSA6 TTT`, `16L XSAall TTT`, `14L XSA6 GPTQ`, `SP8192 HessMix SWA`, `13L XSA4 GPTQ`, `12L EMA Trigram`, `12L Int4 Trigram`, `LoRA TTT`, `11L Int4 QAT`, `SP4096 DepthRecur`, `SP8192 Merge-Conquer`, `MixedQuant SlidingWindow`, `SmearGate OrthoInit`, `10L Int5 MLP`, `Int6 QAT SlidingWindow`, `SlidingWindow Eval`, `Core training scripts`, `Seq2048 FP16 Emb`, `SlidingWindow FP16 Emb`, `Warmdown Quantization`, `Core model modules`, `Lower LR`, `FP16 Embed WD`, `LongContext Seq2048`, `Training Opt Seq4096`, `10L Mixed Precision`, `Naive Baseline`, `Quasi10B SP1024`?**
  _High betweenness centrality (0.188) - this node is a cross-community bridge._
- **Are the 111 inferred relationships involving `rms_norm()` (e.g. with `.forward()` and `.forward()`) actually correct?**
  _`rms_norm()` has 111 INFERRED edges - model-reasoned connections that need verification._
- **Are the 90 inferred relationships involving `train()` (e.g. with `test_model_forward()` and `eval_val()`) actually correct?**
  _`train()` has 90 INFERRED edges - model-reasoned connections that need verification._
- **Are the 82 inferred relationships involving `get()` (e.g. with `test_mixed_quantize_roundtrip()` and `measure_artifact()`) actually correct?**
  _`get()` has 82 INFERRED edges - model-reasoned connections that need verification._
- **Are the 32 inferred relationships involving `compress()` (e.g. with `test_artifact_size()` and `main()`) actually correct?**
  _`compress()` has 32 INFERRED edges - model-reasoned connections that need verification._