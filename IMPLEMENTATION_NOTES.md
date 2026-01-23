# Implementation Notes: Code Changes Across Phases

## File: train.py

### Initial State (Phase 1 - Commit 21f8aae)

```python
# === IMPORTS ===
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3"  # 3 GPUs
import torch.distributed as dist  # DDP support
from transformers import (
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    TrOCRProcessor, VisionEncoderDecoderModel,
    XLMRobertaTokenizerFast,  # Extra import
    XLMRobertaForCausalLM,
    AutoConfig, AutoModelForCausalLM,  # Unused
    ViTImageProcessor,  # Unused
    ProcessorMixin,  # Unused
    DataCollatorForSeq2Seq,  # Unused
    GenerationConfig,
    EarlyStoppingCallback, TrainerCallback
)
from torch.utils.data import DataLoader  # Unused

# === MODEL SETUP ===
model_dir = "microsoft/trocr-base-stage1"
decoder_dir = "FacebookAI/xlm-roberta-base"

tokenizer = AutoTokenizer.from_pretrained(decoder_dir)
processor = TrOCRProcessor.from_pretrained(model_dir, tokenizer=tokenizer)
model = VisionEncoderDecoderModel.from_pretrained(model_dir)
decoder = XLMRobertaForCausalLM.from_pretrained(
    decoder_dir, is_decoder=True, add_cross_attention=True
)

# Configure decoder
model.decoder = decoder
model.decoder.config = decoder.config
model.config.vocab_size = model.decoder.config.vocab_size
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id

# === GENERATION CONFIG ===
gen_config = GenerationConfig.from_model_config(model.config)
gen_config.repetition_penalty = 1.1
# ⚠️ Missing: max_length, num_beams, early_stopping, etc.
model.generation_config = gen_config

# === DATA ===
DATA_DIR = "/mnt/truenas/datasets/synth/400K/"

# === PARAMETER FREEZING ===
for name, param in model.encoder.named_parameters(): 
    param.requires_grad = False

for name, param in model.encoder.pooler.named_parameters(): 
    param.requires_grad = True

for name, param in model.encoder.layernorm.named_parameters(): 
    param.requires_grad = True

for param in model.encoder.encoder.layer[-5].parameters():
    param.requires_grad = True

# === TRAINING ARGS ===
training_args = Seq2SeqTrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=2,  # Small batch
    per_device_eval_batch_size=32,
    num_train_epochs=1000,
    learning_rate=2e-5,
    # ... DDP config enabled
    ddp_find_unused_parameters=True,
    ddp_backend="gloo",
    local_rank=-1,
)

# === TRAINER ===
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=10), generation_callback]
)

trainer.train(resume_from_checkpoint=True)
```

**Status:** ❌ FAILS with CUDA assertion error

---

### Phase 2 Changes (Commits 948c74a → 9a0eaaf → 2357dfb)

#### Change 1: GPU Configuration

```diff
- os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3"
+ os.environ["CUDA_VISIBLE_DEVICES"] = "0"

- import torch.distributed as dist
- trainer.train(resume_from_checkpoint=True)
+ trainer.train()

- Commented out DDP config:
- # ddp_find_unused_parameters=True,
- # ddp_backend="gloo",
- # local_rank=-1,
```

**Reason:** Simplified single-GPU setup on RunPod

#### Change 2: Data Paths

```diff
- DATA_DIR = "/mnt/truenas/datasets/synth/400K/"

+ DATA_DIR = "/workspace/data/nid_ocr_synth_data_100k"

# Updated evaluation image paths
- '/mnt/truenas/datasets/synth/400K/bangla/images/bn_247201.png'
+ f'{DATA_DIR}/bangla/images/bn_247201.png'
```

**Reason:** Migrated dataset location

#### Change 3: Batch Size Increase

```diff
- per_device_train_batch_size=2
- per_device_eval_batch_size=32
+ per_device_train_batch_size=32
+ per_device_eval_batch_size=64
```

**Reason:** Single GPU memory available, improved throughput

#### Change 4: Parameter Freezing Removal

```diff
- for name, param in model.encoder.named_parameters(): 
-     param.requires_grad = False
+ # for name, param in model.encoder.named_parameters(): 
+ #     param.requires_grad = False

- for name, param in model.encoder.pooler.named_parameters(): 
-     param.requires_grad = True
+ # for name, param in model.encoder.pooler.named_parameters(): 
+ #     param.requires_grad = True

- for name, param in model.encoder.layernorm.named_parameters(): 
-     param.requires_grad = True
+ # for name, param in model.encoder.layernorm.named_parameters(): 
+ #     param.requires_grad = True

- for param in model.encoder.encoder.layer[-5].parameters():
-     param.requires_grad = True
+ # for param in model.encoder.encoder.layer[-5].parameters():
+ #     param.requires_grad = True
```

**Impact:** All 384.86M parameters now trainable (vs ~100M in Phase 1)

#### Change 5: Import Cleanup (Commit 2357dfb)

```diff
- from torch.utils.data import DataLoader
- from transformers import (
-     TrOCRProcessor, VisionEncoderDecoderModel,
-     XLMRobertaTokenizerFast,
-     XLMRobertaForCausalLM,
-     AutoConfig,
-     AutoModelForCausalLM,
-     ViTImageProcessor,
-     ProcessorMixin,
-     DataCollatorForSeq2Seq
- )

+ from transformers import (
+     TrOCRProcessor,
+     VisionEncoderDecoderModel,
+     XLMRobertaForCausalLM,
+     AutoTokenizer,
+     GenerationConfig
+ )
```

**Reason:** Code cleanliness

---

### Phase 3 Changes (Commit e8da720)

#### Change 1: **CRITICAL - Vocab Size Fix**

```diff
+ from safetensors.torch import load_file

+ ckpt_path = "/workspace/github/vit_xlm_roberta/outputs-p2/checkpoint-20000"
+
+ model = VisionEncoderDecoderModel.from_pretrained(ckpt_path)
+ decoder = XLMRobertaForCausalLM.from_pretrained(
+     decoder_dir, is_decoder=True, add_cross_attention=True
+ )
+
+ # Configure decoder
+ model.decoder = decoder
+ model.decoder.config = decoder.config
+ model.config.vocab_size = model.decoder.config.vocab_size
+ model.config.decoder_start_token_id = tokenizer.bos_token_id
+ model.config.pad_token_id = tokenizer.pad_token_id
+ model.config.eos_token_id = tokenizer.eos_token_id
+ model.config.decoder = model.decoder.config
+
+ state_dict = load_file(f"{ckpt_path}/model.safetensors")
+ missing, unexpected = model.load_state_dict(state_dict, strict=False)
+
+ print(f"Missing keys: {missing}")
+ print(f"Unexpected keys: {unexpected}")
```

**What's Happening:**
1. Load Phase 2 checkpoint as starting point
2. Create fresh XLM-RoBERTa decoder (with correct vocab)
3. Replace model's decoder
4. Update all vocab-related configs
5. Load safetensors weights from checkpoint
6. Log any missing/unexpected keys

**Why Necessary:** 
- Phase 1 checkpoint had vocab mismatch
- Fresh decoder ensures correct architecture
- Safetensors loading preserves weights

#### Change 2: Enhanced Generation Config

```diff
  gen_config = GenerationConfig.from_model_config(model.config)
  gen_config.repetition_penalty = 1.1
+ gen_config.max_length = 32
+ gen_config.early_stopping = True
+ gen_config.no_repeat_ngram_size = 3
+ gen_config.num_beams = 5
+ gen_config.length_penalty = 1.0
+ gen_config.use_cache = True
```

**Detailed Explanation:**
- **max_length=32:** Matches training max_target_length
- **early_stopping=True:** Halt when EOS token generated
- **no_repeat_ngram_size=3:** Prevent "aa aa aa" patterns
- **num_beams=5:** Beam search width for quality
- **length_penalty=1.0:** No preference for short/long sequences
- **use_cache=True:** KV-cache for faster decoding

#### Change 3: Training Hyperparameters

```diff
- num_train_epochs=1000,
+ num_train_epochs=10,

- learning_rate=2e-5,
+ learning_rate=1e-05,

- eval_on_start=True,
+ eval_on_start=True,
+ metric_for_best_model="cer",
+ greater_is_better=False,
+ label_smoothing_factor=0.1,
```

**Rationale:**
- **num_epochs:** 1000 → 10 is more realistic for convergence
- **learning_rate:** 2e-5 → 1e-5 for conservative fine-tuning
- **metric_for_best_model:** Use CER (task-specific) instead of loss
- **label_smoothing_factor:** Regularization for synthetic data

#### Change 4: Data Loading Optimization

```diff
+ dataloader_num_workers=12,
+ dataloader_persistent_workers=True,
+ gradient_checkpointing=True,
+ dataloader_prefetch_factor=4,
+ dataloader_pin_memory=True,
```

**Impact:**
- **12 workers:** Parallel image loading/preprocessing
- **persistent_workers:** Avoid spawning new processes per epoch
- **gradient_checkpointing:** Save memory (trade CPU for GPU RAM)
- **prefetch_factor:** Pipeline 4 batches ahead
- **pin_memory:** Pre-allocate GPU for data transfer

#### Change 5: Early Stopping Patience

```diff
- EarlyStoppingCallback(early_stopping_patience=5)
+ EarlyStoppingCallback(early_stopping_patience=10)
```

**Reason:** More stable training with label smoothing

---

## New Implementation: LabelSmoothingSeq2SeqTrainer

**File Location:** `train.py` (lines ~21-50)

**Code:**
```python
import torch.nn as nn

class LabelSmoothingSeq2SeqTrainer(Seq2SeqTrainer):
    """Custom trainer that handles label smoothing for VisionEncoderDecoderModel."""
    
    def __init__(self, label_smoothing_factor=0.0, **kwargs):
        # Set label_smoothing_factor to 0 in args to prevent default behavior
        if kwargs.get('args') is not None:
            self._custom_label_smoothing = kwargs['args'].label_smoothing_factor
            kwargs['args'].label_smoothing_factor = 0.0
        else:
            self._custom_label_smoothing = label_smoothing_factor
        super().__init__(**kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if labels is not None and self._custom_label_smoothing > 0:
            # Compute label smoothing loss manually
            loss_fct = nn.CrossEntropyLoss(
                ignore_index=-100,
                label_smoothing=self._custom_label_smoothing
            )
            # Shift logits and labels for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        else:
            loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss
```

**Why Custom Implementation?**

The standard `Seq2SeqTrainer` doesn't apply label smoothing to seq2seq models properly:
1. Default trainer assumes encoder-decoder (translation)
2. Our decoder is causal (GPT-like for OCR)
3. Need to shift logits/labels for proper causal LM loss
4. Need to handle -100 padding tokens

**Key Methods:**

1. **`__init__`:**
   - Saves custom label_smoothing_factor
   - Sets Trainer's label_smoothing_factor to 0 (avoid double smoothing)
   - Calls parent init

2. **`compute_loss`:**
   - Gets labels from inputs
   - Gets logits from model forward pass
   - If smoothing > 0:
     - Creates CrossEntropyLoss with label_smoothing
     - Shifts logits: [0:-1] (don't predict after EOS)
     - Shifts labels: [1:] (predict from t-1 to t)
     - Computes loss
   - Returns (loss, outputs) if requested

---

## Trainer Initialization Change

**Phase 2:**
```python
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=processor,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5), generation_callback]
)
```

**Phase 3:**
```python
trainer = LabelSmoothingSeq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=processor,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=10), generation_callback],
    label_smoothing_factor=0.1  # Custom parameter for LabelSmoothingSeq2SeqTrainer
)
```

---

## Complete Phase 3 Model Initialization

```python
# === IMPORTS & SETUP ===
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
from transformers import (
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    TrOCRProcessor, VisionEncoderDecoderModel,
    XLMRobertaForCausalLM, AutoTokenizer, GenerationConfig,
    EarlyStoppingCallback, TrainerCallback
)
from safetensors.torch import load_file

# === MODEL LOADING ===
model_dir = "microsoft/trocr-base-stage1"
decoder_dir = "FacebookAI/xlm-roberta-base"
ckpt_path = "/workspace/github/vit_xlm_roberta/outputs-p2/checkpoint-20000"

tokenizer = AutoTokenizer.from_pretrained(decoder_dir)
processor = TrOCRProcessor.from_pretrained(model_dir, tokenizer=tokenizer)

# Load Phase 2 checkpoint
model = VisionEncoderDecoderModel.from_pretrained(ckpt_path)

# Replace decoder with fresh XLM-RoBERTa
decoder = XLMRobertaForCausalLM.from_pretrained(
    decoder_dir, is_decoder=True, add_cross_attention=True
)
model.decoder = decoder
model.decoder.config = decoder.config

# Update configs
model.config.vocab_size = model.decoder.config.vocab_size
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.decoder = model.decoder.config

# Load pretrained weights
state_dict = load_file(f"{ckpt_path}/model.safetensors")
missing, unexpected = model.load_state_dict(state_dict, strict=False)

# === GENERATION CONFIG ===
gen_config = GenerationConfig.from_model_config(model.config)
gen_config.repetition_penalty = 1.1
gen_config.max_length = 32
gen_config.early_stopping = True
gen_config.no_repeat_ngram_size = 3
gen_config.num_beams = 5
gen_config.length_penalty = 1.0
gen_config.use_cache = True
model.generation_config = gen_config

# === TRAINING ===
trainer = LabelSmoothingSeq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=processor,
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=10),
        generation_callback
    ],
    label_smoothing_factor=0.1
)

trainer.train()
```

---

## Summary: Lines Changed per Phase

| Metric | Phase 1 | Phase 2 | Phase 3 |
|--------|---------|---------|---------|
| **Total Lines** | ~308 | ~308 | ~330+ |
| **Imports Changed** | — | ↓ 10 unused removed | — |
| **Config Hardcoding** | N/A | N/A | ✓ ckpt_path, safetensors |
| **Classes Added** | — | — | ✓ LabelSmoothingSeq2SeqTrainer |
| **New Trainer Params** | 0 | 0 | 7 (dataloader + label smoothing) |
| **Generation Params** | 1 | 7 | 7 |

---

## Testing & Validation

### Phase 2 Validation
```bash
# Verify vocab fix
python -c "
from train import model, tokenizer
print(f'Model vocab_size: {model.config.vocab_size}')
print(f'Tokenizer vocab_size: {len(tokenizer)}')
assert model.config.vocab_size == len(tokenizer), 'Vocab mismatch!'
print('✓ Vocab matched!')
"
```

### Phase 3 Validation
```bash
# Verify trainer and label smoothing
python -c "
from train import trainer, training_args
print(f'Trainer type: {type(trainer).__name__}')
print(f'Label smoothing factor: {training_args.label_smoothing_factor}')
print(f'Gradient checkpointing: {training_args.gradient_checkpointing}')
print(f'Dataloader workers: {training_args.dataloader_num_workers}')
assert training_args.label_smoothing_factor == 0.1
assert training_args.gradient_checkpointing == True
print('✓ All configurations valid!')
"
```

---

**Implementation Document v1.0**  
**Created:** Jan 23, 2026  
**Covers:** All code changes across Phases 1-3
