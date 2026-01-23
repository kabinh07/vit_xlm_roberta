# Phase-by-Phase Summary & Changes

## Phase 1: Initial Setup (Jan 16-22)

### Configuration
- **Model:** ViT (encoder) + XLM-RoBERTa (decoder)
- **Dataset:** 1M synthetic images (500K Bengali + 500K English)
- **Training Steps:** 76,000
- **GPU:** 3x (later switched to 1x)

### Key Hyperparameters
```
batch_size: 32
learning_rate: 2e-5
num_epochs: 1000 (theoretical)
early_stopping_patience: 5 → 10
```

### Model Freezing
- Encoder: FROZEN (except pooler, layernorm, last 5 layers)
- Decoder: ALL TRAINABLE

### Generation Config
```
min config:
- repetition_penalty: 1.1
- default num_beams: 1
```

### Status
❌ CUDA Assertion Error (vocab size mismatch: 50K vs 250K)

---

## Phase 2: Bug Fix & Stabilization (Jan 22-23)

### Critical Changes
✅ **Vocab Size Fix:**
- Replaced decoder with fresh XLM-RoBERTa instance
- Loaded Phase 1 weights with safetensors
- Config updated: vocab_size = 250,265

### Enhanced Generation Config
```python
gen_config.max_length = 32
gen_config.early_stopping = True
gen_config.no_repeat_ngram_size = 3
gen_config.num_beams = 5
gen_config.length_penalty = 1.0
gen_config.use_cache = True
```

### Parameter Changes
- Removed all encoder freezing (all parameters trainable)
- Cleaned up unused imports

### Training
- Steps: 20,000 (resumed from P1)
- Total: 96,000 cumulative steps
- Status: ✅ Resumed successfully

---

## Phase 3: Advanced Training (Jan 23+)

### Architectural Innovation
✨ **LabelSmoothingSeq2SeqTrainer** - Custom trainer with label smoothing

```python
class LabelSmoothingSeq2SeqTrainer(Seq2SeqTrainer):
    - Custom compute_loss() with label smoothing
    - CrossEntropyLoss(label_smoothing=0.1)
    - Proper sequence shifting for causal LM
```

### Hyperparameter Tuning

| Parameter | Phase 2 | Phase 3 | Why? |
|-----------|---------|---------|------|
| num_epochs | 1000 | 10 | Realistic convergence target |
| learning_rate | 2e-5 | 1e-5 | More conservative fine-tuning |
| label_smoothing | — | 0.1 | Reduce overconfidence |
| metric_for_best_model | — | "cer" | Task-specific optimization |
| gradient_checkpointing | — | True | Memory optimization |
| dataloader_workers | — | 12 | Parallel I/O |
| dataloader_pin_memory | — | True | Fast GPU transfer |

### New Optimizations
- Gradient checkpointing (memory efficiency)
- Parallel data loading (12 workers)
- GPU memory pinning
- CER as primary metric
- Increased early stopping patience (10)

### Status
🟡 In Progress (Phase 3 started)

---

## Git Commit History

```
e8da720 - Running and configured for phase 3
          ├── Added LabelSmoothingSeq2SeqTrainer
          ├── Reduced num_epochs: 1000 → 10
          ├── Reduced learning_rate: 2e-5 → 1e-5
          ├── Added label_smoothing_factor: 0.1
          ├── Added gradient_checkpointing: True
          ├── Added dataloader_num_workers: 12
          └── Added dataloader_pin_memory: True

2357dfb - [UPDATE] clean import files
          └── Removed unused imports

9a0eaaf - [UPDATE] for runpod
          ├── Switched GPU: 3 → 1
          ├── Vocab fix: 50K → 250K ✓
          ├── Enhanced generation config
          ├── Removed encoder freezing
          └── Updated data paths

948c74a - REMOVED gpu export from sh
          └── Minor: early_stopping_patience adjustment

21f8aae - Initial
          └── Phase 1 baseline setup
```

---

## Bug Summary

### Phase 1 Issue: CUDA Assertion Error
```
Error: /pytorch/aten/src/ATen/native/cuda/Indexing.cu:1553: 
       indexSelectLargeIndex: Assertion `srcIndex < srcSelectDimSize` failed
```

**Root Cause:**
- Checkpoint decoder vocab: 50,265 tokens (TrOCR)
- XLM-RoBERTa tokenizer: 250,265 tokens
- Token IDs > 50K → out-of-bounds array access

**Solution (Phase 2):**
```python
decoder = XLMRobertaForCausalLM.from_pretrained(decoder_dir, ...)
model.decoder = decoder
state_dict = load_file(ckpt_path + "/model.safetensors")
model.load_state_dict(state_dict, strict=False)
```

---

## Key Metrics Tracked

### Training Metrics
- **Loss:** Training loss, Validation loss
- **Learning:** Learning rate, Gradient norm
- **Performance:** CER (Character Error Rate), WER (Word Error Rate)

### Generation Quality
- **Repetition Count:** Tokens repeated per sample
- **Repetition Rate:** % of samples with repetitions
- **Avg Length:** Average generated sequence length

### Logged Destinations
- TensorBoard: `./runs/` (scalars + text summaries)
- CSV: `generation_metrics.csv` (per eval_step)
- Console: Loss, CER/WER, generation samples

---

## Why These Changes Matter

### Label Smoothing (Phase 3)
- Prevents overconfidence on synthetic data
- Softens hard one-hot targets
- Improves generalization to real OCR

### Learning Rate Reduction (Phase 3)
- Protects pretrained encoder knowledge
- Finer-grained fine-tuning
- Reduces catastrophic forgetting

### Parallel Data Loading (Phase 3)
- Eliminates I/O bottleneck
- Ensures GPU utilization at 32 batch size
- 12 workers for efficient prefetching

### Gradient Checkpointing (Phase 3)
- Trades computation for memory
- Enables larger batches or longer sequences
- No accuracy loss (recomputes on backward)

---

## Checklist: What's Implemented?

- ✅ Phase 1: Baseline setup + initial training (76K steps)
- ✅ Phase 2: Vocab fix + enhanced generation (20K steps)
- ✅ Phase 3: Label smoothing + optimizations (in progress)
- ✅ Git tracking: 5 commits with clear progression
- ✅ Monitoring: TensorBoard + CSV logging
- ✅ Documentation: This file + R_D_DOCUMENTATION.md

---

## Next Steps

1. **Monitor Phase 3 Training:**
   - Track CER improvement
   - Watch for label smoothing impact
   - Check GPU memory usage

2. **Post-Training Analysis:**
   - Compare CER across phases
   - Analyze generation quality
   - Evaluate on real OCR data

3. **Deployment Preparation:**
   - Export best checkpoint
   - Create inference script
   - Benchmark latency/throughput

---

**Last Updated:** Jan 23, 2026  
**Phase Status:** Phase 3 In Progress  
**Next Documentation:** Post-training analysis report
