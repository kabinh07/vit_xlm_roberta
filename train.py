import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_SHM_DISABLE"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3"

import os
import jiwer
import json
from PIL import Image
import mlflow
import mlflow.pytorch
import tarfile
import math

import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter

from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, XLMRobertaForCausalLM, AutoTokenizer, GenerationConfig
from transformers import EarlyStoppingCallback, TrainerCallback
from datasets import load_dataset

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

from safetensors.torch import load_file

import random

import unicodedata

import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and processor
model_dir = "microsoft/trocr-base-stage1"
decoder_dir = "FacebookAI/xlm-roberta-base"
ckpt_path = os.path.abspath("outputs-p3/checkpoint-18000")
hf_dir = "kavinh07/vit-xlmroberta-nid-ocr"
DATA_DIR = "kavinh07/nid-synth-200k-ocr"

tokenizer = AutoTokenizer.from_pretrained(decoder_dir)
processor = TrOCRProcessor.from_pretrained(model_dir, tokenizer=tokenizer)
model = VisionEncoderDecoderModel.from_pretrained(hf_dir)
# model = VisionEncoderDecoderModel.from_pretrained(model_dir)
# decoder = XLMRobertaForCausalLM.from_pretrained(decoder_dir, is_decoder=True, add_cross_attention=True)

model.decoder.config.is_decoder = True
model.decoder.config.add_cross_attention = True

# # Configure decoder
# model.decoder = decoder
# model.decoder.config = decoder.config
# model.config.vocab_size = model.decoder.config.vocab_size
# model.config.decoder_start_token_id = tokenizer.bos_token_id
# model.config.pad_token_id = tokenizer.pad_token_id
# model.config.eos_token_id = tokenizer.eos_token_id
# model.config.decoder = model.decoder.config

# state_dict = load_file(f"{ckpt_path}/model.safetensors")
# missing, unexpected = model.load_state_dict(state_dict, strict=False)

# print(f"Missing keys: {missing}")
# print(f"Unexpected keys: {unexpected}")
with open("best_params.json", "r") as f:
    best_params = json.load(f)

gen_config = GenerationConfig.from_model_config(model.config)
gen_config.repetition_penalty = best_params["repetition_penalty"]
gen_config.max_length = 64
gen_config.early_stopping = True
gen_config.no_repeat_ngram_size = best_params["no_repeat_ngram_size"]
gen_config.num_beams = best_params["num_beams"]
gen_config.length_penalty = best_params["length_penalty"]
gen_config.use_cache = True

model.generation_config = gen_config

def prepare_sharded_dataset(hf_data_dir, shard_dir="shards", samples_per_shard=2000):
    """
    Download HuggingFace dataset and create shards.
    Returns path to shard directory.
    """
    os.makedirs(shard_dir, exist_ok=True)
    
    # Check if shards already exist
    existing_shards = [f for f in os.listdir(shard_dir) if f.endswith(".tar")]
    if existing_shards:
        print(f"Using existing shards: {len(existing_shards)} shards found")
        return shard_dir
    
    # Download dataset from HuggingFace if not present
    if not os.path.exists("hf_dataset"):
        print(f"Downloading dataset {hf_data_dir}...")
        dataset = load_dataset(hf_data_dir)
        dataset.save_to_disk("hf_dataset")
    
    img_dir = "hf_dataset/train/images" if os.path.exists("hf_dataset/train") else "hf_dataset/images"
    lbl_dir = "hf_dataset/train/labels" if os.path.exists("hf_dataset/train") else "hf_dataset/labels"
    
    # Create shards from the dataset
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not os.path.exists(lbl_dir):
        raise FileNotFoundError(f"Label directory not found: {lbl_dir}")
    
    files = sorted(os.listdir(img_dir))
    num_shards = math.ceil(len(files) / samples_per_shard)
    
    print(f"Creating {num_shards} shards from {len(files)} files...")
    for shard_id in range(num_shards):
        start = shard_id * samples_per_shard
        end = min((shard_id + 1) * samples_per_shard, len(files))
        
        shard_path = f"{shard_dir}/shard-{shard_id:05d}.tar"
        
        with tarfile.open(shard_path, "w") as tar:
            for fname in files[start:end]:
                base = os.path.splitext(fname)[0]
                
                img_path = os.path.join(img_dir, fname)
                lbl_path = os.path.join(lbl_dir, base + ".txt")
                
                if os.path.exists(img_path) and os.path.exists(lbl_path):
                    tar.add(img_path, arcname=f"{base}.jpg")
                    tar.add(lbl_path, arcname=f"{base}.txt")
        
        if (shard_id + 1) % 10 == 0:
            print(f"  Created {shard_id + 1}/{num_shards} shards")
    
    return shard_dir


class ShardedOCRDataset(Dataset):
    """Load data from tar shards instead of raw files."""
    def __init__(self, shard_dir, processor, max_target_length=32):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_target_length = max_target_length
        self.shard_dir = shard_dir
        self.samples = []
        
        # Index all samples from shards
        shard_files = sorted([f for f in os.listdir(shard_dir) if f.endswith(".tar")])
        
        for shard_file in shard_files:
            shard_path = os.path.join(shard_dir, shard_file)
            with tarfile.open(shard_path, "r") as tar:
                for member in tar.getmembers():
                    if member.name.endswith(".jpg"):
                        base_name = member.name[:-4]  # Remove .jpg
                        self.samples.append((shard_path, base_name))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        shard_path, base_name = self.samples[idx]
        
        # Open tar file and extract image and label
        with tarfile.open(shard_path, "r") as tar:
            # Read image
            img_member = tar.getmember(f"{base_name}.jpg")
            img_file = tar.extractfile(img_member)
            image = Image.open(img_file).convert("RGB")
            
            # Read label
            lbl_member = tar.getmember(f"{base_name}.txt")
            lbl_file = tar.extractfile(lbl_member)
            text = lbl_file.read().decode("utf-8")
        
        text = unicodedata.normalize("NFKC", text)
        pixel_values = self.processor(image, return_tensors="pt")["pixel_values"]
        tokenized = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt"
        )
        
        labels = tokenized.input_ids.squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        encoding = {
            "pixel_values": pixel_values.squeeze(),
            "labels": labels
        }
        return encoding

# Prepare sharded dataset
shard_dir = prepare_sharded_dataset(DATA_DIR, shard_dir="shards", samples_per_shard=2000)

data = ShardedOCRDataset(shard_dir="shards", processor=processor)

print(f"Sample of dataset:\n{data[0]}")

train_size = int(0.99 * len(data))
val_size = len(data) - train_size

train_dataset, val_dataset = random_split(data, [train_size, val_size])
print(f"Train Size: {len(train_dataset)}\nTest Size: {len(val_dataset)}")


# def compute_metrics(pred):
#     pred_ids = pred.predictions
#     label_ids = pred.label_ids

#     # Replace -100 with pad_token_id
#     pred_ids[pred_ids == -100] = processor.tokenizer.pad_token_id
#     label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

#     pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
#     label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

#     cer = jiwer.cer(label_str, pred_str)
#     wer = jiwer.wer(label_str, pred_str)

#     return {"cer": cer, "wer": wer}

# # for name, param in model.decoder.named_parameters(): 
# #     print(name, param)

# # for name, param in model.decoder.named_parameters(): 
# #     param.requires_grad = False

# # for name, param in model.decoder.roberta.embeddings.named_parameters():
# #     param.requires_grad = True

# # for name, param in model.encoder.pooler.named_parameters(): 
# #     param.requires_grad = True

# # for name, param in model.encoder.layernorm.named_parameters(): 
# #     param.requires_grad = True

# # for param in model.encoder.encoder.layer[-5].parameters():
# #     param.requires_grad = True


# print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1000000:.2f} million")

# class GenerationCallback(TrainerCallback):
#     def __init__(self, processor, model, eval_images, eval_texts, tokenizer, output_dir="trocr-xlm-roberta-finetuned-synth-400k"):
#         self.processor = processor
#         self.model = model
#         self.eval_images = eval_images
#         self.eval_texts = eval_texts
#         self.tokenizer = tokenizer
#         self.output_dir = output_dir
#         self.writer = None
    
#     def on_evaluate(self, args, state, control, trainer=None, **kwargs):
#         current_step = state.global_step
        
#         # Initialize SummaryWriter if not already done
#         if self.writer is None:
#             self.writer = SummaryWriter(log_dir=self.output_dir)
        
#         # Metrics to track step-wise
#         metrics_log = {"step": current_step}
#         repetition_count = 0
#         avg_generated_length = 0
        
#         # Build text summary for TensorBoard
#         text_summary = f"## Generation Test - Step {current_step}\n\n"
        
        
#         self.model.eval()
        
#         for i, (image, text) in enumerate(zip(self.eval_images, self.eval_texts)):
#             pil_image = Image.open(image).convert("RGB")
#             pixel_values = self.processor(images=pil_image, return_tensors="pt").pixel_values.to(self.model.device)

#             with open(text, "r", encoding="utf-8") as f:
#                 true_text = f.read()
            
#             with torch.no_grad():
#                 generated_ids = self.model.generate(pixel_values)
            
#             generated_text = self.processor.decode(generated_ids[0], skip_special_tokens=True)
#             tokens = generated_ids[0].tolist()
            
#             # Add to text summary for TensorBoard
#             text_summary += f"### Sample {i+1}\n"
#             text_summary += f"**Ground Truth:** {true_text}\n\n"
#             text_summary += f"**Generated:** {generated_text}\n\n"

#             # Check repetition
#             has_repetition = False
#             if len(tokens) > 3 and len(set(tokens[1:-1])) == 1:
#                 has_repetition = True
#                 repetition_count += 1
#                 text_summary += f"⚠️ **WARNING:** REPETITION DETECTED!\n\n"
            
#             # Track generated length
#             avg_generated_length += len(tokens)
        
#         # Calculate averages and log to TensorBoard
#         avg_generated_length = avg_generated_length / len(self.eval_images)
#         repetition_rate = (repetition_count / len(self.eval_images)) * 100
        
#         # Add summary metrics to text
#         text_summary += f"\n### Metrics Summary\n"
#         text_summary += f"- **Repetition Count:** {repetition_count}/{len(self.eval_images)}\n"
#         text_summary += f"- **Repetition Rate:** {repetition_rate:.1f}%\n"
#         text_summary += f"- **Average Length:** {avg_generated_length:.1f} tokens\n"
        
#         # Log text to TensorBoard
#         if self.writer is not None:
#             self.writer.add_text("generation/samples", text_summary, current_step)
#             self.writer.flush()
        
#         # Log metrics to TensorBoard via trainer
#         tensorboard_metrics = {
#             "generation/repetition_count": repetition_count,
#             "generation/repetition_rate": repetition_rate,
#             "generation/avg_length": avg_generated_length,
#         }
        
#         if trainer is not None:
#             trainer.log(tensorboard_metrics)
        
#         # Also save to CSV for easy tracking
#         csv_path = "generation_metrics.csv"
#         csv_metrics = {"step": current_step, **tensorboard_metrics}
#         df = pd.DataFrame([csv_metrics])
#         if os.path.exists(csv_path):
#             df.to_csv(csv_path, mode="a", header=False, index=False)
#         else:
#             df.to_csv(csv_path, mode="w", index=False)

# eval_images = [
#     f'{DATA_DIR}/images/bn_img_4383.png',
#     f'{DATA_DIR}/images/bn_img_4856.png',
#     f'{DATA_DIR}/images/en_img_11559.png',
#     f'{DATA_DIR}/images/en_img_14207.png'
# ]

# eval_texts = [
#     f'{DATA_DIR}/labels/bn_img_4383.txt',
#     f'{DATA_DIR}/labels/bn_img_4856.txt',
#     f'{DATA_DIR}/labels/en_img_11559.txt',
#     f'{DATA_DIR}/labels/en_img_14207.txt'
# ]

# generation_callback = GenerationCallback(
#     processor=processor,
#     model=model,
#     eval_images=eval_images,
#     eval_texts=eval_texts,
#     tokenizer=processor.tokenizer,
#     output_dir="./runs"
# )

# if __name__ == "__main__":
#     training_args = Seq2SeqTrainingArguments(
#         output_dir="./outputs",
#         per_device_train_batch_size=32,
#         per_device_eval_batch_size=64,
#         num_train_epochs=1000,
#         fp16=False,
#         save_steps=1000,
#         logging_steps=100,
#         eval_steps=1000,
#         report_to="tensorboard",
#         logging_dir="./runs",
#         save_total_limit=2,
#         predict_with_generate=True,
#         gradient_accumulation_steps=1,
#         learning_rate=1e-05,
#         lr_scheduler_type="cosine",
#         warmup_steps=100,
#         load_best_model_at_end=True,
#         eval_strategy="steps",
#         weight_decay=0.005,
#         eval_on_start=True,
#         metric_for_best_model="cer",
#         greater_is_better=False,
#         ddp_find_unused_parameters=True,
#         dataloader_num_workers=12,
#         dataloader_persistent_workers=True,
#         gradient_checkpointing=True,
#         dataloader_prefetch_factor=4,
#         dataloader_pin_memory=True,
#         ddp_backend="gloo",
#         local_rank=-1,
#         deepspeed="ds_config.json",
#         hub_model_id="kavinh07/vit-xlmroberta-nid-ocr",
#         push_to_hub=True,
#     )

#     trainer = Seq2SeqTrainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=val_dataset,
#         processing_class=processor,
#         compute_metrics=compute_metrics,
#         callbacks=[EarlyStoppingCallback(early_stopping_patience=20), generation_callback]
#     )
#     try:    
#         # Train the model
#         trainer.train()
#         print("Training completed")

#     except Exception as e:
#         print(f"Training interrupted: {e}")

#     finally:
#         # Start MLflow run
#         with mlflow.start_run():
#             # Log dataset information
#             mlflow.log_param("dataset_path", DATA_DIR)
#             mlflow.log_param("train_size", train_size)
#             mlflow.log_param("val_size", val_size)
#             mlflow.log_param("train_percentage", 0.99)
            
#             # Log model information
#             mlflow.log_param("encoder_model", model_dir)
#             mlflow.log_param("decoder_model", decoder_dir)
#             mlflow.log_param("checkpoint_path", ckpt_path)
            
#             # Log generation config parameters
#             mlflow.log_param("repetition_penalty", model.generation_config.repetition_penalty)
#             mlflow.log_param("no_repeat_ngram_size", model.generation_config.no_repeat_ngram_size)
#             mlflow.log_param("num_beams", model.generation_config.num_beams)
#             mlflow.log_param("length_penalty", model.generation_config.length_penalty)
#             mlflow.log_param("max_length", model.generation_config.max_length)
            
#             # Log training hyperparameters
#             mlflow.log_param("per_device_train_batch_size", training_args.per_device_train_batch_size)
#             mlflow.log_param("per_device_eval_batch_size", training_args.per_device_eval_batch_size)
#             mlflow.log_param("num_train_epochs", training_args.num_train_epochs)
#             mlflow.log_param("learning_rate", training_args.learning_rate)
#             mlflow.log_param("lr_scheduler_type", training_args.lr_scheduler_type)
#             mlflow.log_param("warmup_steps", training_args.warmup_steps)
#             mlflow.log_param("weight_decay", training_args.weight_decay)
#             mlflow.log_param("gradient_accumulation_steps", training_args.gradient_accumulation_steps)
#             mlflow.log_param("eval_strategy", training_args.eval_strategy)
#             mlflow.log_param("eval_steps", training_args.eval_steps)
#             mlflow.log_param("save_steps", training_args.save_steps)
#             mlflow.log_param("logging_steps", training_args.logging_steps)

#             # Log total trainable parameters
#             total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000
#             mlflow.log_param("total_trainable_parameters_millions", total_params)
# else:
#     # For importing in other scripts, create trainer without DDP settings
#     training_args = Seq2SeqTrainingArguments(
#         output_dir="./outputs",
#         per_device_train_batch_size=4,
#         per_device_eval_batch_size=8,
#         num_train_epochs=10,
#         fp16=False,
#         save_steps=1000,
#         logging_steps=100,
#         eval_steps=1000,
#         report_to="tensorboard",
#         logging_dir="./runs",
#         save_total_limit=2,
#         push_to_hub=False,
#         predict_with_generate=True,
#         gradient_accumulation_steps=4,
#         learning_rate=1e-05,
#         lr_scheduler_type="cosine",
#         warmup_steps=100,
#         load_best_model_at_end=True,
#         eval_strategy="steps", 
#         weight_decay=0.005,
#         eval_on_start=True,
#         metric_for_best_model="cer",
#         greater_is_better=False,
#         deepspeed="ds_config.json",
#     )
    
#     trainer = Seq2SeqTrainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=val_dataset,
#         processing_class=processor,
#         compute_metrics=compute_metrics,
#         callbacks=[EarlyStoppingCallback(early_stopping_patience=10), generation_callback]
#     )