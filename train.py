import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import os
import jiwer
from PIL import Image

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter

from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, XLMRobertaTokenizerFast, XLMRobertaForCausalLM, XLMRobertaForCausalLM
from transformers import AutoConfig, AutoModelForCausalLM, ViTImageProcessor, AutoTokenizer, ProcessorMixin, DataCollatorForSeq2Seq
from transformers import GenerationConfig
from transformers import EarlyStoppingCallback, TrainerCallback

import random

import unicodedata

import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and processor
model_dir = "microsoft/trocr-base-stage1"
decoder_dir = "FacebookAI/xlm-roberta-base"

tokenizer = AutoTokenizer.from_pretrained(decoder_dir)
processor = TrOCRProcessor.from_pretrained(model_dir, tokenizer=tokenizer)
model = VisionEncoderDecoderModel.from_pretrained(model_dir)
decoder = XLMRobertaForCausalLM.from_pretrained(decoder_dir, is_decoder=True, add_cross_attention=True)

# Configure decoder
model.decoder = decoder
model.decoder.config = decoder.config
model.config.vocab_size = model.decoder.config.vocab_size
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id

gen_config = GenerationConfig.from_model_config(model.config)
gen_config.repetition_penalty = 1.1
model.generation_config = gen_config

DATA_DIR = "/workspace/data/nid_ocr_synth_data_100k"

class OCRDataset(Dataset):
    def __init__(self, data_dir, processor, max_target_length=32, small_dataset_size=None):
        self.bn_image_dir = os.path.join(data_dir, "bangla/images")
        self.en_image_dir = os.path.join(data_dir, "english/images")
        self.bn_images = os.listdir(self.bn_image_dir)
        self.en_images = os.listdir(self.en_image_dir)
        self.images = self.__get_total_images()
        self.bn_label_dir = os.path.join(data_dir, "bangla/labels")
        self.en_label_dir = os.path.join(data_dir, "english/labels")
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_target_length = max_target_length

    def __get_total_images(self):
        images = []
        self.en_images = [image for image in self.en_images if image.strip() != ""]
        self.bn_images = [image for image in self.bn_images if image.strip() != ""]
        print(f"Total english images: {len(self.en_images)} and bangla images: {len(self.bn_images)}")
        for en, bn in zip(self.en_images, self.bn_images): 
            images.extend([bn, en])
        return images
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if idx % 2 == 0:
            image_path = os.path.join(self.bn_image_dir, self.images[idx])
            label_path = os.path.join(self.bn_label_dir, self.images[idx].split(".")[0]+".txt")
        else:
            image_path = os.path.join(self.en_image_dir, self.images[idx])
            label_path = os.path.join(self.en_label_dir, self.images[idx].split(".")[0]+".txt")
        image = Image.open(image_path).convert("RGB")
        with open(label_path, "r", encoding="utf-8") as f:
            text = f.read()
        if idx % 2 == 0:
            text = unicodedata.normalize("NFC", text)
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

data = OCRDataset(DATA_DIR, processor)

print(f"Sample of dataset:\n{data[0]}")

train_size = int(0.999 * len(data))
val_size = len(data) - train_size

train_dataset, val_dataset = random_split(data, [train_size, val_size])
print(f"Train Size: {len(train_dataset)}\nTest Size: {len(val_dataset)}")


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad_token_id
    pred_ids[pred_ids == -100] = processor.tokenizer.pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    cer = jiwer.cer(label_str, pred_str)
    wer = jiwer.wer(label_str, pred_str)

    return {"cer": cer, "wer": wer}

# for name, param in model.encoder.named_parameters(): 
#     if not param.requires_grad:
#         print(name)

# for name, param in model.encoder.named_parameters(): 
#     param.requires_grad = False

# for name, param in model.encoder.pooler.named_parameters(): 
#     param.requires_grad = True

# for name, param in model.encoder.layernorm.named_parameters(): 
#     param.requires_grad = True

# for param in model.encoder.encoder.layer[-5].parameters():
#     param.requires_grad = True


print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1000000:.2f} million")

class GenerationCallback(TrainerCallback):
    def __init__(self, processor, model, eval_images, eval_texts, tokenizer, output_dir="trocr-xlm-roberta-finetuned-synth-400k"):
        self.processor = processor
        self.model = model
        self.eval_images = eval_images
        self.eval_texts = eval_texts
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.writer = None
    
    def on_evaluate(self, args, state, control, trainer=None, **kwargs):
        current_step = state.global_step
        
        # Initialize SummaryWriter if not already done
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.output_dir)
        
        # Metrics to track step-wise
        metrics_log = {"step": current_step}
        repetition_count = 0
        avg_generated_length = 0
        
        # Build text summary for TensorBoard
        text_summary = f"## Generation Test - Step {current_step}\n\n"
        
        
        self.model.eval()
        
        for i, (image, text) in enumerate(zip(self.eval_images, self.eval_texts)):
            pil_image = Image.open(image).convert("RGB")
            pixel_values = self.processor(images=pil_image, return_tensors="pt").pixel_values.to(self.model.device)

            with open(text, "r", encoding="utf-8") as f:
                true_text = f.read()
            
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values)
            
            generated_text = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            tokens = generated_ids[0].tolist()
            
            # Add to text summary for TensorBoard
            text_summary += f"### Sample {i+1}\n"
            text_summary += f"**Ground Truth:** {true_text}\n\n"
            text_summary += f"**Generated:** {generated_text}\n\n"

            # Check repetition
            has_repetition = False
            if len(tokens) > 3 and len(set(tokens[1:-1])) == 1:
                has_repetition = True
                repetition_count += 1
                text_summary += f"⚠️ **WARNING:** REPETITION DETECTED!\n\n"
            
            # Track generated length
            avg_generated_length += len(tokens)
        
        # Calculate averages and log to TensorBoard
        avg_generated_length = avg_generated_length / len(self.eval_images)
        repetition_rate = (repetition_count / len(self.eval_images)) * 100
        
        # Add summary metrics to text
        text_summary += f"\n### Metrics Summary\n"
        text_summary += f"- **Repetition Count:** {repetition_count}/{len(self.eval_images)}\n"
        text_summary += f"- **Repetition Rate:** {repetition_rate:.1f}%\n"
        text_summary += f"- **Average Length:** {avg_generated_length:.1f} tokens\n"
        
        # Log text to TensorBoard
        if self.writer is not None:
            self.writer.add_text("generation/samples", text_summary, current_step)
            self.writer.flush()
        
        # Log metrics to TensorBoard via trainer
        tensorboard_metrics = {
            "generation/repetition_count": repetition_count,
            "generation/repetition_rate": repetition_rate,
            "generation/avg_length": avg_generated_length,
        }
        
        if trainer is not None:
            trainer.log(tensorboard_metrics)
        
        # Also save to CSV for easy tracking
        csv_path = "generation_metrics.csv"
        csv_metrics = {"step": current_step, **tensorboard_metrics}
        df = pd.DataFrame([csv_metrics])
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode="a", header=False, index=False)
        else:
            df.to_csv(csv_path, mode="w", index=False)

eval_images = [
    f'{DATA_DIR}/bangla/images/bn_247201.png',
    f'{DATA_DIR}/bangla/images/bn_247202.png',
    f'{DATA_DIR}/english/images/en_178613.png',
    f'{DATA_DIR}/english/images/en_178614.png'
]

eval_texts = [
    f'{DATA_DIR}/bangla/labels/bn_247201.txt',
    f'{DATA_DIR}/bangla/labels/bn_247202.txt',
    f'{DATA_DIR}/english/labels/en_178613.txt',
    f'{DATA_DIR}/english/labels/en_178614.txt'
]

generation_callback = GenerationCallback(
    processor=processor,
    model=model,
    eval_images=eval_images,
    eval_texts=eval_texts,
    tokenizer=processor.tokenizer,
    output_dir="./runs"
)

training_args = Seq2SeqTrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=1000,
    fp16=torch.cuda.is_available(),
    save_steps=1000,
    logging_steps=100,
    eval_steps=1000,
    report_to="tensorboard",
    logging_dir="./runs",
    save_total_limit=2,
    push_to_hub=False,
    predict_with_generate=True,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    load_best_model_at_end=True,
    eval_strategy="steps", 
    weight_decay=0.005,
    eval_on_start=True,
    # ddp_find_unused_parameters=True,
    # ddp_backend="gloo",
    # local_rank=-1,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=processor,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5), generation_callback]
)

if __name__ == "__main__":
    try:
        trainer.train()
    except Exception as e:
        print(f"Training interrupted: {e}")