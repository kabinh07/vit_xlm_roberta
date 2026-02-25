import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_SHM_DISABLE"] = "1"

import jiwer
import json
from PIL import Image
import mlflow
import mlflow.pytorch
import tarfile
import math
import numpy as np

import torch
from torch.utils.data import Dataset, random_split
from torch.utils.tensorboard import SummaryWriter

from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    MBartForCausalLM,
    MBart50Tokenizer,           # slow tokenizer — avoids tiktoken/SentencePiece conversion error
    GenerationConfig,
)
from transformers import EarlyStoppingCallback, TrainerCallback
from datasets import load_dataset

import pandas as pd
import unicodedata

device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# CONFIGURATION
# ============================================================
trocr_dir   = "microsoft/trocr-base-stage1"  # source of pretrained ViT encoder
decoder_dir = "facebook/mbart-large-50"
hf_dir      = "kavinh07/nid-ocr-vit-mbart50"
DATA_DIR    = "kavinh07/synth-200k-ocr"
TEST_DIR    = "./test_data"

LANG_CODE_BN = "bn_IN"
LANG_CODE_EN = "en_XX"

# ============================================================
# TOKENIZER
# ============================================================
tokenizer = MBart50Tokenizer.from_pretrained(
    decoder_dir,
    src_lang=LANG_CODE_BN,
    tgt_lang=LANG_CODE_BN,
)

# TrOCRProcessor = ViT feature extractor + a tokenizer
processor = TrOCRProcessor.from_pretrained(trocr_dir, tokenizer=tokenizer)

# ============================================================
# BUILD MODEL
#
# WHY NOT from_encoder_decoder_pretrained():
#   TrOCR is itself a VisionEncoderDecoderModel. Passing it as
#   the encoder_pretrained_model_name_or_path causes AutoModel
#   to reject its VisionEncoderDecoderConfig.
#
# SOLUTION:
#   1. Load TrOCR, extract just its ViT encoder.
#   2. Load mBART-50 as a causal decoder.
#   3. Pass both to VisionEncoderDecoderModel(encoder=, decoder=).
#      This constructor detects hidden_size mismatch (768 vs 1024)
#      and auto-creates enc_to_dec_proj correctly.
# ============================================================

# Step 1 — extract ViT encoder from TrOCR
trocr_model = VisionEncoderDecoderModel.from_pretrained(trocr_dir)
vit_encoder = trocr_model.encoder   # ViT, hidden_size = 768
del trocr_model                      # free memory immediately

# Step 2 — load mBART-50 decoder
mbart_decoder = MBartForCausalLM.from_pretrained(
    decoder_dir,
    is_decoder=True,
    add_cross_attention=True,
)

# Step 3 — assemble model; enc_to_dec_proj (768->1024) created automatically
model = VisionEncoderDecoderModel(
    encoder=vit_encoder,
    decoder=mbart_decoder,
)

# Step 4 — language token IDs
BN_TOKEN_ID = tokenizer.lang_code_to_id[LANG_CODE_BN]
EN_TOKEN_ID = tokenizer.lang_code_to_id[LANG_CODE_EN]

# Step 5 — set config on BOTH model.config and model.generation_config
#           (setting only model.config triggers a deprecation warning in v4.x)
for cfg in (model.config, model.generation_config):
    cfg.decoder_start_token_id = BN_TOKEN_ID
    cfg.forced_bos_token_id    = BN_TOKEN_ID
    cfg.pad_token_id           = tokenizer.pad_token_id
    cfg.eos_token_id           = tokenizer.eos_token_id

model.config.vocab_size = model.decoder.config.vocab_size

# Sanity check
print(f"enc_to_dec_proj : {model.enc_to_dec_proj}")
print(f"Encoder hidden  : {vit_encoder.config.hidden_size}")
print(f"Decoder hidden  : {mbart_decoder.config.hidden_size}")

# ============================================================
# GENERATION CONFIG
# ============================================================
with open("best_params.json", "r") as f:
    best_params = json.load(f)

model.generation_config.repetition_penalty = best_params.get("repetition_penalty", 1.3)
model.generation_config.max_length         = 64
model.generation_config.early_stopping     = True
model.generation_config.use_cache          = True

# ============================================================
# FREEZE ENCODER  (keep last transformer block trainable)
# ============================================================
for param in model.encoder.parameters():
    param.requires_grad = False

for param in model.encoder.encoder.layer[-1].parameters():
    param.requires_grad = True

# enc_to_dec_proj is randomly initialised — must stay trainable
for param in model.enc_to_dec_proj.parameters():
    param.requires_grad = True

print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")


# ============================================================
# DATASET UTILITIES
# ============================================================

def prepare_sharded_dataset(data_source, shard_dir="shards", samples_per_shard=2000, max_shards=None):
    os.makedirs(shard_dir, exist_ok=True)
    existing_shards = sorted([f for f in os.listdir(shard_dir) if f.endswith(".tar")])

    if existing_shards:
        shards = existing_shards[:max_shards] if max_shards else existing_shards
        print(f"Using {len(shards)} existing shards")
        return shard_dir

    if os.path.isdir(data_source):
        import shutil
        source_shards = sorted([f for f in os.listdir(data_source) if f.endswith(".tar")])
        if source_shards:
            if max_shards:
                source_shards = source_shards[:max_shards]
            for sf in source_shards:
                src = os.path.join(data_source, sf)
                dst = os.path.join(shard_dir, sf)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
            print(f"Loaded {len(source_shards)} shards from {data_source}")
            return shard_dir

    print(f"Downloading dataset: {data_source} ...")
    dataset     = load_dataset(data_source, split="train")
    num_samples = len(dataset)
    num_shards  = math.ceil(num_samples / samples_per_shard)

    if max_shards:
        num_shards  = min(num_shards, max_shards)
        dataset     = dataset.select(range(min(num_shards * samples_per_shard, len(dataset))))
        num_samples = len(dataset)

    import io
    print(f"Creating {num_shards} shards ...")
    for shard_id in range(num_shards):
        start         = shard_id * samples_per_shard
        end           = min(start + samples_per_shard, num_samples)
        shard_path    = os.path.join(shard_dir, f"shard-{shard_id:05d}.tar")
        shard_dataset = dataset.select(range(start, end))

        with tarfile.open(shard_path, "w") as tar:
            for idx, sample in enumerate(shard_dataset):
                image = sample["jpg"]
                text  = sample["txt"]

                img_buf = io.BytesIO()
                image.save(img_buf, format="PNG")
                img_data = img_buf.getvalue()

                ti      = tarfile.TarInfo(name=f"sample_{start+idx:06d}.png")
                ti.size = len(img_data)
                tar.addfile(ti, io.BytesIO(img_data))

                tb      = text.encode("utf-8")
                tt      = tarfile.TarInfo(name=f"sample_{start+idx:06d}.txt")
                tt.size = len(tb)
                tar.addfile(tt, io.BytesIO(tb))

        if (shard_id + 1) % max(1, num_shards // 10) == 0:
            print(f"  {shard_id+1}/{num_shards} shards done")

    return shard_dir


class ShardedOCRDataset(Dataset):
    def __init__(self, shard_dir, processor, max_target_length=64, max_samples=None):
        self.processor         = processor
        self.tokenizer         = processor.tokenizer
        self.max_target_length = max_target_length
        self.samples           = []

        shard_files = sorted([f for f in os.listdir(shard_dir) if f.endswith(".tar")])
        for sf in shard_files:
            with tarfile.open(os.path.join(shard_dir, sf), "r") as tar:
                for member in tar.getmembers():
                    if member.name.endswith((".jpg", ".png")):
                        base = member.name.rsplit(".", 1)[0]
                        self.samples.append((os.path.join(shard_dir, sf), base))
                        if max_samples and len(self.samples) >= max_samples:
                            break
            if max_samples and len(self.samples) >= max_samples:
                break

        if max_samples:
            self.samples = self.samples[:max_samples]

        print(f"Loaded {len(self.samples)} samples from {len(shard_files)} shards")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        shard_path, base = self.samples[idx]

        with tarfile.open(shard_path, "r") as tar:
            image = None
            for ext in (".jpg", ".png"):
                try:
                    image = Image.open(tar.extractfile(tar.getmember(f"{base}{ext}"))).convert("RGB")
                    break
                except KeyError:
                    continue
            if image is None:
                raise FileNotFoundError(f"No image for {base}")

            text = tar.extractfile(
                tar.getmember(f"{base}.txt")
            ).read().decode("utf-8").strip()

        text         = unicodedata.normalize("NFKC", text)
        pixel_values = self.processor(image, return_tensors="pt")["pixel_values"].squeeze()

        tokenized = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        )

        labels = tokenized.input_ids.squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}


# ============================================================
# DATA SETUP
# ============================================================
TEST_MODE   = False
MAX_SHARDS  = None
MAX_SAMPLES = None

shard_dir = prepare_sharded_dataset(
    data_source=DATA_DIR,
    shard_dir="shards",
    samples_per_shard=2000,
    max_shards=MAX_SHARDS if TEST_MODE else None,
)

data = ShardedOCRDataset(
    shard_dir=shard_dir,
    processor=processor,
    max_target_length=64,
    max_samples=MAX_SAMPLES if TEST_MODE else None,
)

train_size = int(0.99 * len(data))
val_size   = len(data) - train_size
train_dataset, val_dataset = random_split(data, [train_size, val_size])
print(f"Train: {train_size}  Val: {val_size}")


# ============================================================
# METRICS
# ============================================================
def compute_metrics(pred):
    pred_ids  = pred.predictions
    label_ids = pred.label_ids

    pred_ids[pred_ids == -100]   = tokenizer.pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str  = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    return {
        "cer": jiwer.cer(label_str, pred_str),
        "wer": jiwer.wer(label_str, pred_str),
    }


# ============================================================
# GENERATION CALLBACK
# ============================================================
class GenerationCallback(TrainerCallback):
    def __init__(self, processor, model, eval_images, eval_texts, tokenizer, output_dir="./outputs/runs"):
        self.processor   = processor
        self.model       = model
        self.eval_images = eval_images
        self.eval_texts  = eval_texts
        self.tokenizer   = tokenizer
        self.output_dir  = output_dir
        self.writer      = None

    def encoder_heatmap(self, hidden_state):
        patches = hidden_state.cpu().detach().numpy()[0, 1:, :]
        return np.linalg.norm(patches, axis=-1).reshape(24, 24)

    def on_evaluate(self, args, state, control, trainer=None, **kwargs):
        step = state.global_step
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.output_dir)

        repetition_count, total_len = 0, 0
        text_summary = f"## Generation Test - Step {step}\n\n"
        self.model.eval()

        for i, (img_path, txt_path) in enumerate(zip(self.eval_images, self.eval_texts)):
            pil   = Image.open(img_path).convert("RGB")
            pv    = self.processor(images=pil, return_tensors="pt").pixel_values.to(self.model.device)
            truth = open(txt_path, encoding="utf-8").read()

            with torch.no_grad():
                outs = self.model.generate(
                    pv,
                    forced_bos_token_id=BN_TOKEN_ID,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                )

            hm = self.encoder_heatmap(outs.encoder_hidden_states[-1])
            hm_up = torch.nn.functional.interpolate(
                torch.tensor(hm).unsqueeze(0).unsqueeze(0),
                size=(pil.height, pil.width), mode="bilinear", align_corners=False
            )[0, 0].numpy()

            if self.writer:
                img_np = np.array(pil).astype(np.float32) / 255.0
                h_min, h_max = hm_up.min(), hm_up.max()
                h_norm = (hm_up - h_min) / (h_max - h_min + 1e-8)
                import matplotlib.cm as cm
                hm_color = cm.get_cmap("viridis")(h_norm)[:, :, :3].astype(np.float32)
                overlay  = img_np * 0.5 + hm_color * 0.5
                self.writer.add_image(f"gen/overlay_{i+1}", np.transpose(overlay, (2,0,1)), step)
                self.writer.add_image(f"gen/image_{i+1}",   np.transpose(img_np,  (2,0,1)), step)

            gen_text = self.tokenizer.decode(outs.sequences[0], skip_special_tokens=True)
            tokens   = outs.sequences[0].tolist()
            total_len += len(tokens)

            text_summary += f"### Sample {i+1}\n**GT:** {truth}\n\n**Pred:** {gen_text}\n\n"
            if len(tokens) > 3 and len(set(tokens[1:-1])) == 1:
                repetition_count += 1
                text_summary += "⚠️ REPETITION DETECTED\n\n"

        n    = max(len(self.eval_images), 1)
        avg  = total_len / n
        rate = repetition_count / n * 100

        text_summary += f"**Repetitions:** {repetition_count}/{n}  **Rate:** {rate:.1f}%  **Avg len:** {avg:.1f}\n"
        if self.writer:
            self.writer.add_text("gen/samples", text_summary, step)
            self.writer.flush()

        metrics = {"gen/repetition_rate": rate, "gen/avg_length": avg}
        if trainer:
            trainer.log(metrics)

        csv_path = "generation_metrics.csv"
        df = pd.DataFrame([{"step": step, **metrics}])
        df.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)


# ============================================================
# EVAL SAMPLE EXTRACTION
# ============================================================
def extract_eval_samples_from_shards(shard_dir, num_samples=4):
    import tempfile
    eval_dir = tempfile.mkdtemp(prefix="eval_")
    imgs, txts, count = [], [], 0

    for sf in sorted(f for f in os.listdir(shard_dir) if f.endswith(".tar")):
        if count >= num_samples:
            break
        with tarfile.open(os.path.join(shard_dir, sf), "r") as tar:
            for m in tar.getmembers():
                if not m.name.endswith((".jpg", ".png")):
                    continue
                base = m.name.rsplit(".", 1)[0]
                try:
                    txt_m = tar.getmember(f"{base}.txt")
                except KeyError:
                    continue

                ip = os.path.join(eval_dir, f"{base}.png")
                with open(ip, "wb") as f:
                    f.write(tar.extractfile(m).read())

                tp = os.path.join(eval_dir, f"{base}.txt")
                with open(tp, "w", encoding="utf-8") as f:
                    f.write(tar.extractfile(txt_m).read().decode("utf-8"))

                imgs.append(ip)
                txts.append(tp)
                count += 1
                if count >= num_samples:
                    break

    return imgs, txts


# ============================================================
# EVAL SAMPLES SETUP
# ============================================================
if os.path.isdir(TEST_DIR):
    eval_images = [
        f"{TEST_DIR}/bn_000000.png",   f"{TEST_DIR}/bn_img_22.png",
        f"{TEST_DIR}/bn_img_57.png",   f"{TEST_DIR}/bn_img_2517.png",
        f"{TEST_DIR}/en_img_5866.png", f"{TEST_DIR}/en_img_6353.png",
        f"{TEST_DIR}/en_img_8000.png",
    ]
    eval_texts = [
        f"{TEST_DIR}/bn_000000.txt",   f"{TEST_DIR}/bn_img_22.txt",
        f"{TEST_DIR}/bn_img_57.txt",   f"{TEST_DIR}/bn_img_2517.txt",
        f"{TEST_DIR}/en_img_5866.txt", f"{TEST_DIR}/en_img_6353.txt",
        f"{TEST_DIR}/en_img_8000.txt",
    ]
else:
    print("Extracting eval samples from shards...")
    eval_images, eval_texts = extract_eval_samples_from_shards(shard_dir, num_samples=4)

generation_callback = None
if eval_images and eval_texts:
    generation_callback = GenerationCallback(
        processor=processor, model=model,
        eval_images=eval_images, eval_texts=eval_texts,
        tokenizer=tokenizer, output_dir="./outputs/runs",
    )


# ============================================================
# TRAINING
# ============================================================
if __name__ == "__main__":
    training_args = Seq2SeqTrainingArguments(
        output_dir="./outputs",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=1000,
        fp16=False,
        save_steps=500,
        logging_steps=100,
        eval_steps=500,
        report_to="tensorboard",
        logging_dir="./outputs/runs",
        save_total_limit=2,
        predict_with_generate=True,
        gradient_accumulation_steps=4,
        learning_rate=3e-6,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_steps=1000,
        max_grad_norm=1.0,
        load_best_model_at_end=True,
        eval_strategy="steps",
        eval_on_start=True,
        metric_for_best_model="eval_loss",
        ddp_find_unused_parameters=True,
        dataloader_num_workers=12,
        dataloader_persistent_workers=False,
        gradient_checkpointing=True,
        dataloader_prefetch_factor=4,
        dataloader_pin_memory=True,
        deepspeed="ds_config.json",
        hub_model_id=hf_dir,
        push_to_hub=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor,
        compute_metrics=compute_metrics,
        callbacks=(
            [EarlyStoppingCallback(early_stopping_patience=10)]
            + ([generation_callback] if generation_callback else [])
        ),
    )

    try:
        ckpts = [f for f in os.listdir(training_args.output_dir) if f.startswith("checkpoint")] \
                if os.path.exists(training_args.output_dir) else []
        if ckpts:
            print("Resuming from checkpoint...")
            trainer.train(resume_from_checkpoint=True)
        else:
            print("Starting from scratch...")
            trainer.train()
        print("Training complete.")

    except Exception as e:
        print(f"Training interrupted: {e}")
        raise

    finally:
        trainer.push_to_hub()
        with mlflow.start_run():
            mlflow.log_params({
                "decoder_model": decoder_dir,
                "dataset_path":  DATA_DIR,
                "train_size":    train_size,
                "val_size":      val_size,
                "model_name":    hf_dir,
                "lang_code_bn":  LANG_CODE_BN,
                "lang_code_en":  LANG_CODE_EN,
                "learning_rate": training_args.learning_rate,
                "weight_decay":  training_args.weight_decay,
                "warmup_steps":  training_args.warmup_steps,
                "grad_accum":    training_args.gradient_accumulation_steps,
                "trainable_M":   round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 2),
            })