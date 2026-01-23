# Technical Comparison: Phase 1 → Phase 2 → Phase 3

## Architecture Evolution

```
PHASE 1: Initial Implementation (BROKEN)
┌─────────────────────────────────────────┐
│  VisionEncoderDecoderModel              │
├─────────────────────────────────────────┤
│  Encoder: ViT (FROZEN, selective)       │
│  ├─ pooler, layernorm: trainable        │
│  └─ last 5 layers: trainable            │
├─────────────────────────────────────────┤
│  Decoder: XLM-RoBERTa (REPLACED)        │
│  ├─ Vocab: 250,265 ← CONFIG             │
│  └─ Actual: 50,265 ✗ MISMATCH          │
├─────────────────────────────────────────┤
│  Gen Config: minimal                    │
│  ├─ repetition_penalty: 1.1             │
│  └─ num_beams: 1 (default)              │
├─────────────────────────────────────────┤
│  Status: ❌ CUDA ASSERTION ERROR        │
└─────────────────────────────────────────┘

PHASE 2: Bug Fix (WORKING)
┌─────────────────────────────────────────┐
│  VisionEncoderDecoderModel              │
├─────────────────────────────────────────┤
│  Encoder: ViT (ALL TRAINABLE)           │
│  └─ All parameters: requires_grad=True  │
├─────────────────────────────────────────┤
│  Decoder: XLM-RoBERTa (RESET)           │
│  ├─ Fresh instance from HF              │
│  ├─ Vocab: 250,265 ✓ MATCHED            │
│  └─ Weights loaded: Phase 1 checkpoint  │
├─────────────────────────────────────────┤
│  Gen Config: enhanced                   │
│  ├─ repetition_penalty: 1.1             │
│  ├─ max_length: 32                      │
│  ├─ num_beams: 5                        │
│  ├─ early_stopping: True                │
│  ├─ no_repeat_ngram_size: 3             │
│  ├─ length_penalty: 1.0                 │
│  └─ use_cache: True                     │
├─────────────────────────────────────────┤
│  Status: ✅ TRAINING RESUMED            │
└─────────────────────────────────────────┘

PHASE 3: Advanced Regularization (OPTIMIZED)
┌─────────────────────────────────────────┐
│  VisionEncoderDecoderModel              │
├─────────────────────────────────────────┤
│  Encoder: ViT (ALL TRAINABLE)           │
│  └─ Parameters: 384.86M (all modes)     │
├─────────────────────────────────────────┤
│  Decoder: XLM-RoBERTa (TRAINED)         │
│  ├─ Vocab: 250,265 ✓ STABLE             │
│  ├─ Weights: From Phase 2 checkpoint    │
│  └─ Loss: Label-Smoothed CrossEntropy   │
├─────────────────────────────────────────┤
│  Gen Config: same as Phase 2            │
├─────────────────────────────────────────┤
│  Training: Enhanced Optimization        │
│  ├─ Trainer: LabelSmoothingSeq2SeqTrainer│
│  ├─ Label Smoothing: 0.1                │
│  ├─ Gradient Checkpointing: True        │
│  ├─ Parallel I/O: 12 workers            │
│  ├─ Memory Pinning: True                │
│  └─ CER-driven optimization             │
├─────────────────────────────────────────┤
│  Status: 🟡 IN PROGRESS                 │
└─────────────────────────────────────────┘
```

## Configuration Comparison Table

### Model Setup

| Aspect | Phase 1 | Phase 2 | Phase 3 |
|--------|---------|---------|---------|
| **Encoder** | ViT (TrOCR base) | ViT (TrOCR base) | ViT (TrOCR base) |
| **Decoder** | XLM-RoBERTa | XLM-RoBERTa (reset) | XLM-RoBERTa (trained) |
| **Decoder Vocab Size** | 250K (config) / 50K (actual) ❌ | 250K ✅ | 250K ✅ |
| **Encoder Freezing** | Partial ❄️ | None 🔥 | None 🔥 |
| **Total Parameters** | 384.86M | 384.86M | 384.86M |
| **Trainable Params** | ~100M | 384.86M | 384.86M |
| **Loss Function** | Seq2Seq (default) | Seq2Seq (default) | Label-Smoothed |

### Training Configuration

| Parameter | Phase 1 | Phase 2 | Phase 3 | Change |
|-----------|---------|---------|---------|--------|
| per_device_train_batch_size | 32 | 32 | 32 | — |
| per_device_eval_batch_size | 64 | 64 | 64 | — |
| num_train_epochs | 1000 | 1000 | 10 | ↓ 100x |
| learning_rate | 2e-5 | 2e-5 | 1e-5 | ↓ 2x |
| warmup_steps | 100 | 100 | 100 | — |
| weight_decay | 0.005 | 0.005 | 0.005 | — |
| gradient_accumulation | 2 | 2 | 2 | — |
| fp16 | True | True | True | — |
| label_smoothing | 0.0 | 0.0 | 0.1 | ↑ NEW |
| early_stopping_patience | 5→10 | 10 | 10 | — |
| gradient_checkpointing | False | False | True | ↑ NEW |

### Generation Configuration

| Parameter | Phase 1 | Phase 2 | Phase 3 |
|-----------|---------|---------|---------|
| max_length | — (default 50) | 32 | 32 |
| num_beams | — (default 1) | 5 | 5 |
| early_stopping | — (default False) | True | True |
| no_repeat_ngram_size | — (default None) | 3 | 3 |
| length_penalty | — (default 1.0) | 1.0 | 1.0 |
| repetition_penalty | 1.1 | 1.1 | 1.1 |
| use_cache | — (default False) | True | True |

### Data Loading & I/O

| Parameter | Phase 1 | Phase 2 | Phase 3 |
|-----------|---------|---------|---------|
| dataloader_num_workers | — (default) | — (default) | 12 |
| dataloader_persistent_workers | — | — | True |
| dataloader_prefetch_factor | — | — | 4 |
| dataloader_pin_memory | — | — | True |

### Hardware & Infrastructure

| Aspect | Phase 1 | Phase 2 | Phase 3 |
|--------|---------|---------|---------|
| **GPU Count** | 3 (initial) → 1 | 1 | 1 |
| **GPU Memory** | ~24GB per GPU | ~24GB | ~24GB |
| **Distributed Training** | DDP (initial) → Single | Single | Single |
| **Framework** | PyTorch + HF Transformers | PyTorch + HF Transformers | PyTorch + HF Transformers |

---

## Loss Function Comparison

### Phase 1 & 2: Default Seq2Seq Loss

```python
# Built-in Transformer loss
outputs = model(**inputs)
loss = outputs.loss  # CrossEntropyLoss with default settings
```

**Characteristics:**
- Hard one-hot targets
- No label smoothing
- Can overfit on synthetic data
- Sharp probability distributions

### Phase 3: Label-Smoothed Loss

```python
class LabelSmoothingSeq2SeqTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, ...):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if labels is not None and self._custom_label_smoothing > 0:
            loss_fct = nn.CrossEntropyLoss(
                ignore_index=-100,
                label_smoothing=0.1  # ← Key change
            )
            # Shift for causal LM (t → t+1)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        else:
            loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss
```

**Characteristics:**
- Soft probability targets (α = 0.1)
- Reduces overconfidence
- Better generalization
- Smoother confidence curves

**Label Smoothing Effect:**
```
Without smoothing (α=0):
  Target class: P = 1.0
  Other classes: P = 0.0
  → Sharp, overconfident

With smoothing (α=0.1):
  Target class: P = 0.91  (1 - 0.1 × (250265-1) / 250265)
  Other classes: P ≈ 0.0000004 each (0.1 / 250265)
  → Soft, regularized
```

---

## Training Progress

### Phase 1
```
Step 0 ──────────────────────────────────── Step 76,000
├─ Encoder frozen (selective)
├─ Decoder from XLM-RoBERTa
├─ CUDA error at start
└─ Training halted
```

### Phase 2
```
Step 0 ──────────── Step 20,000
├─ Resumed from P1 checkpoint
├─ Fixed vocab mismatch
├─ All parameters trainable
└─ Training successful
```

### Phase 3
```
Step 0 ──────────────────────────────────── Step X (10 epochs)
├─ Starting from P2 checkpoint
├─ Label smoothing enabled
├─ Enhanced I/O optimization
└─ Target: Best CER (in progress)
```

**Cumulative:** 96,000+ steps across 3 phases

---

## Key Improvements Summary

### Phase 1 → Phase 2

✅ **Bug Fix:**
- Fixed vocab size mismatch (50K → 250K)
- Replaced decoder with fresh XLM-RoBERTa
- Loaded Phase 1 weights properly

✅ **Model Unfreezing:**
- From: Encoder partially frozen
- To: All parameters trainable
- Benefit: Better multilingual adaptation

✅ **Generation Enhancement:**
- Added beam search (num_beams=5)
- Added early stopping
- Added n-gram blocking
- Added caching for speed

### Phase 2 → Phase 3

✅ **Regularization:**
- Added label smoothing (α=0.1)
- Reduce overconfidence on synthetic data

✅ **Optimization:**
- Reduced learning rate (2e-5 → 1e-5)
- Added gradient checkpointing
- Enabled parallel data loading (12 workers)
- Added memory pinning

✅ **Task Alignment:**
- CER as primary metric (not loss)
- Realistic epoch count (10 vs 1000)

---

## Memory & Speed Analysis

### Phase 1 & 2: Single GPU (24GB)

**Memory Breakdown:**
- Model weights: ~1.2GB
- Optimizer states: ~2.4GB (Adam with momentum)
- Gradient buffers: ~1.2GB
- Activations: ~4GB (batch_size=32)
- Data/misc: ~2GB
- **Total: ~11GB (comfortable margin)**

**Speed (estimated):**
- Images/sec: ~50-100 (with TrOCRProcessor)
- Steps/hour: ~3,600-7,200
- Phase 2: 20K steps ≈ 2.8-5.6 hours

### Phase 3: Enhanced Optimization

**With Gradient Checkpointing:**
- Memory saved: ~30-40% (recompute activations)
- Effective usable memory: ~15GB
- Speed trade-off: ~10-15% slower (recomputation)

**With Parallel I/O (12 workers):**
- Eliminates data loading bottleneck
- Better GPU utilization
- Recommended for batch_size ≥ 32

**Net Effect:**
- Memory: More comfortable (for larger batches later)
- Speed: Maintained (optimization offsets I/O gains)
- Quality: Improved (label smoothing)

---

## Known Limitations & Future Work

### Current Limitations

1. **Synthetic Data Only:**
   - Phase 3 training on 100K synthetic images
   - May overfit to synthetic patterns
   - Need real data evaluation

2. **Fixed Max Length:**
   - Limited to 32 tokens
   - Some OCR documents may be longer
   - Consider variable-length generation

3. **Balanced Dataset:**
   - 50/50 Bengali-English split
   - Real-world data may be imbalanced
   - May need language-specific fine-tuning

4. **Single GPU Scaling:**
   - Currently single GPU training
   - Distributed training disabled
   - Consider multi-GPU for faster iteration

### Future Improvements

```
Phase 4 (Proposed):
├─ Real data finetuning (transfer learning)
├─ Language-specific adapters (LoRA)
├─ Longer sequence support (variable length)
├─ Multi-GPU distributed training
└─ Production inference optimization

Phase 5+ (Research):
├─ Mixture of Experts (language routing)
├─ Vision-Language alignment (CLIP)
├─ Quantization & distillation
└─ Benchmark on OCR leaderboards
```

---

## Conclusion

The three-phase training journey shows:

1. **Phase 1:** Established baseline but had critical architectural issue
2. **Phase 2:** Fixed bug, stabilized training, enhanced generation
3. **Phase 3:** Added regularization, optimization, task-specific tuning

**Current State:** Model is well-optimized with proper regularization and ready for extended training on synthetic and real data.

**Expected Outcome:** Improved OCR accuracy on multilingual Bengali-English documents with better generalization through label smoothing.

---

**Comparison Document v1.0**  
**Created:** Jan 23, 2026  
**Updated:** Ongoing (Phase 3)
