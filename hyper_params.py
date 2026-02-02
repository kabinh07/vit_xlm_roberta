import os
# Use single GPU to avoid NCCL errors with multi-GPU setup
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import optuna
import torch
from train import val_dataset, tokenizer, model, processor, device
from transformers import GenerationConfig
from torch.utils.data import Subset, DataLoader
from PIL import Image
import jiwer

def objective(trial):
    # Define the search space
    gen_config = GenerationConfig(
        num_beams=trial.suggest_int("num_beams", 2, 8),
        repetition_penalty=trial.suggest_float("repetition_penalty", 1.0, 1.5),
        length_penalty=trial.suggest_float("length_penalty", 0.6, 1.4),
        no_repeat_ngram_size=trial.suggest_int("no_repeat_ngram_size", 2, 4),
        decoder_start_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        max_length=32,
    )
    
    # Use a small subset of validation data to avoid OOM
    eval_subset = Subset(val_dataset, range(min(50, len(val_dataset))))
    
    # Clear GPU cache before evaluation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Unwrap model if it's wrapped in DataParallel
    eval_model = model
    if isinstance(model, torch.nn.DataParallel):
        eval_model = model.module
    
    eval_model.eval()
    eval_model.to(device)
    
    all_pred_texts = []
    all_label_texts = []
    
    # Simple evaluation loop without trainer
    with torch.no_grad():
        for batch_idx, sample in enumerate(eval_subset):
            try:
                pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
                labels = sample["labels"]
                
                # Generate predictions
                generated_ids = eval_model.generate(pixel_values, generation_config=gen_config)
                pred_text = processor.decode(generated_ids[0], skip_special_tokens=True)
                
                # Decode labels
                labels[labels == -100] = tokenizer.pad_token_id
                label_text = processor.decode(labels, skip_special_tokens=True)
                
                all_pred_texts.append(pred_text)
                all_label_texts.append(label_text)
            except Exception as e:
                print(f"Error in sample {batch_idx}: {e}")
                continue
    
    # Compute CER
    if all_pred_texts and all_label_texts:
        cer = jiwer.cer(all_label_texts, all_pred_texts)
        return cer
    else:
        return 1.0  # Return worst score if evaluation failed

# Create and run the study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

import json
with open("best_params.json", "w") as f:
    json.dump(study.best_params, f, indent=4)
