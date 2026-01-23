# Vision Transformer + XLM-Roberta OCR Training Documentation

## Overview
This project fine-tunes a VisionEncoderDecoderModel (ViT + TrOCR) with an XLM-Roberta decoder for OCR tasks in Bangla and English. Training was conducted in three phases, each with distinct goals and configurations.

---

## Phase 1: Initial Training
- **Goal:** Establish baseline OCR performance using synthetic data (400K samples).
- **Model:**
  - Encoder: ViT (TrOCR base)
  - Decoder: XLM-Roberta
- **Key Steps:**
  - Created custom `OCRDataset` for Bangla/English images and labels.
  - Configured decoder and generation settings (repetition penalty, token IDs).
  - Used TensorBoard and CSV logging for generation metrics (CER, WER, repetition).
  - Early stopping and generation callback implemented.
  - Training run: 1000 epochs, batch size 2 (train), 32 (eval).
  - Output: `outputs/`, `runs/` directories.

---

## Phase 2: Data & Training Improvements
- **Goal:** Scale up training and optimize for distributed/cloud environments (RunPod).
- **Changes:**
  - Switched to a new dataset path: `/workspace/data/nid_ocr_synth_data_100k`.
  - Increased batch sizes: 32 (train), 64 (eval).
  - Reduced epochs to 10 for faster iteration.
  - Updated training script to use single GPU (`CUDA_VISIBLE_DEVICES=0`).
  - Disabled distributed training code for simplicity.
  - Enhanced generation config: max length, early stopping, no repeat n-gram, beam search.
  - Added label smoothing and best model metric (CER).
  - Output: `outputs-p1/`, `runs-p1/` directories.

---

## Phase 3: Model Checkpointing & Final Tuning
- **Goal:** Load and fine-tune from previous checkpoints, improve generalization.
- **Changes:**
  - Model loaded from safetensors checkpoint (`outputs-p2/checkpoint-20000/model.safetensors`).
  - Used `safetensors` for secure state dict loading.
  - Generation config further tuned (beam search, length penalty, cache).
  - Early stopping patience increased to 10.
  - Output: `outputs-p2/`, `runs-p2/` directories.
  - Added notebook for model inspection: `notebooks/check_model.ipynb`.

---

## How to Reproduce Training
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run training:**
   ```bash
   bash run_training.sh
   ```
   - Training logs: `train.log`
   - Checkpoints: `outputs*/checkpoint-*/`
   - TensorBoard: `runs*/`

## Notebooks
- `notebooks/check_model.ipynb`: Inspect model checkpoints and generation config.

## Outputs
- Model checkpoints, configs, and metrics are saved in `outputs*` and `generation_metrics*.csv`.

## References
- [TrOCR](https://huggingface.co/microsoft/trocr-base-stage1)
- [XLM-Roberta](https://huggingface.co/FacebookAI/xlm-roberta-base)
- [safetensors](https://github.com/huggingface/safetensors)

---

## Contact
Maintainer: Kabin Hasan (<kanchon@polygontech.xyz>)
