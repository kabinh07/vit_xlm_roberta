import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"]  = "1"
os.environ["NCCL_SHM_DISABLE"] = "1"

import io
import jiwer
import json
import math
import tarfile
import unicodedata
import numpy as np

from PIL import Image

import torch
from torch.utils.data import Dataset, random_split
from torch.utils.tensorboard import SummaryWriter

from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    MBartForCausalLM,
    MBart50Tokenizer,
)
from transformers import EarlyStoppingCallback, TrainerCallback
from datasets import load_dataset

import mlflow
import mlflow.pytorch
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# CONFIGURATION
# ============================================================
trocr_dir    = "microsoft/trocr-base-stage1"
decoder_dir  = "facebook/mbart-large-50"
hf_dir       = "kavinh07/nid-ocr-vit-mbart50-2"
DATA_DIR     = "kavinh07/synth-200k-ocr"
TEST_DIR     = "./test_data"
LANG_CODE_BN = "bn_IN"
LANG_CODE_EN = "en_XX"

# 128 tokens comfortably covers even the longest NID address fields.
MAX_TARGET_LENGTH = 128

# ============================================================
# LANGUAGE DETECTION
# Used consistently in both dataset tokenization AND inference.
# ============================================================
def is_bangla(text: str) -> bool:
    bangla = sum(1 for c in text if "\u0980" <= c <= "\u09FF")
    return (bangla / max(len(text.strip()), 1)) >= 0.3


# ============================================================
# TOKENIZER & PROCESSOR
# ============================================================
tokenizer = MBart50Tokenizer.from_pretrained(
    decoder_dir,
    src_lang=LANG_CODE_BN,
    tgt_lang=LANG_CODE_BN,
)
processor = TrOCRProcessor.from_pretrained(trocr_dir, tokenizer=tokenizer)

# ============================================================
# BUILD MODEL
# Cannot use from_encoder_decoder_pretrained(trocr_dir, ...) because
# TrOCR is itself a VisionEncoderDecoderModel and AutoModel rejects it.
# Extract ViT from TrOCR, load mBART-50 separately, pass both to
# VisionEncoderDecoderModel() which auto-creates enc_to_dec_proj
# (Linear 768->1024).
# ============================================================
trocr_base    = VisionEncoderDecoderModel.from_pretrained(trocr_dir)
vit_encoder   = trocr_base.encoder
del trocr_base

mbart_decoder = MBartForCausalLM.from_pretrained(
    decoder_dir,
    is_decoder=True,
    add_cross_attention=True,
)

model = VisionEncoderDecoderModel(encoder=vit_encoder, decoder=mbart_decoder)

BN_TOKEN_ID = tokenizer.lang_code_to_id[LANG_CODE_BN]   # 250028
EN_TOKEN_ID = tokenizer.lang_code_to_id[LANG_CODE_EN]   # 250004

# FIX (double BOS): use EOS as a neutral decoder seed.
# Previously decoder_start_token_id=BN_TOKEN_ID caused a duplicate
# language token: model seeded with bn_IN, then label also started
# with bn_IN (prepended by tokenizer). Now:
#   - Decoder is seeded with </s> (neutral, no language bias)
#   - Label starts with first real text token (language token stripped)
#   - At inference, forced_bos_token_id per-sample handles language routing
model.config.decoder_start_token_id            = tokenizer.eos_token_id
model.config.pad_token_id                      = tokenizer.pad_token_id
model.config.eos_token_id                      = tokenizer.eos_token_id
model.config.vocab_size                        = model.decoder.config.vocab_size
model.generation_config.decoder_start_token_id = tokenizer.eos_token_id
model.generation_config.pad_token_id           = tokenizer.pad_token_id
model.generation_config.eos_token_id           = tokenizer.eos_token_id
# forced_bos_token_id NOT set globally — applied per-sample at inference

print(f"enc_to_dec_proj : {model.enc_to_dec_proj}")
print(f"Encoder hidden  : {vit_encoder.config.hidden_size}")
print(f"Decoder hidden  : {mbart_decoder.config.hidden_size}")

# ============================================================
# GENERATION CONFIG
# repetition_penalty softly discourages repeat tokens.
# no_repeat_ngram_size=0: DISABLED — hard-blocks legitimate digit
#   patterns like 7474398146 (the "74" bigram appears twice).
# num_beams=4: beam search for better long-sequence decoding.
# ============================================================
with open("best_params.json", "r") as f:
    best_params = json.load(f)

model.generation_config.repetition_penalty   = best_params.get("repetition_penalty", 1.3)
model.generation_config.no_repeat_ngram_size = 0
model.generation_config.num_beams            = 4
model.generation_config.length_penalty       = 1.0
model.generation_config.max_length           = MAX_TARGET_LENGTH
model.generation_config.early_stopping       = True
model.generation_config.use_cache            = True

# ============================================================
# FREEZE STRATEGY
# Last 3 encoder blocks unfrozen — NID card images are domain-shifted
# from TrOCR pretraining data, need encoder adaptation.
# enc_to_dec_proj always trainable (randomly initialised).
# Full decoder is trainable.
# ============================================================
for param in model.encoder.parameters():
    param.requires_grad = False

for layer in model.encoder.encoder.layer[-3:]:
    for param in layer.parameters():
        param.requires_grad = True

for param in model.enc_to_dec_proj.parameters():
    param.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M total")


# ============================================================
# DATASET
# ============================================================

def prepare_sharded_dataset(
    data_source, shard_dir="shards", samples_per_shard=2000, max_shards=None
):
    os.makedirs(shard_dir, exist_ok=True)
    existing = sorted(f for f in os.listdir(shard_dir) if f.endswith(".tar"))

    if existing:
        use = existing[:max_shards] if max_shards else existing
        print(f"Using {len(use)} existing shards")
        return shard_dir

    if os.path.isdir(data_source):
        import shutil
        srcs = sorted(f for f in os.listdir(data_source) if f.endswith(".tar"))
        if srcs:
            if max_shards:
                srcs = srcs[:max_shards]
            for sf in srcs:
                dst = os.path.join(shard_dir, sf)
                if not os.path.exists(dst):
                    shutil.copy2(os.path.join(data_source, sf), dst)
            print(f"Copied {len(srcs)} shards from {data_source}")
            return shard_dir

    print(f"Downloading: {data_source} ...")
    dataset     = load_dataset(data_source, split="train")
    num_samples = len(dataset)
    num_shards  = math.ceil(num_samples / samples_per_shard)

    if max_shards:
        num_shards  = min(num_shards, max_shards)
        dataset     = dataset.select(range(min(num_shards * samples_per_shard, len(dataset))))
        num_samples = len(dataset)

    print(f"Creating {num_shards} shards from {num_samples} samples ...")
    for shard_id in range(num_shards):
        start         = shard_id * samples_per_shard
        end           = min(start + samples_per_shard, num_samples)
        shard_path    = os.path.join(shard_dir, f"shard-{shard_id:05d}.tar")
        shard_dataset = dataset.select(range(start, end))

        with tarfile.open(shard_path, "w") as tar:
            for idx, sample in enumerate(shard_dataset):
                img_buf = io.BytesIO()
                sample["jpg"].save(img_buf, format="PNG")
                img_data = img_buf.getvalue()

                ti = tarfile.TarInfo(name=f"sample_{start+idx:06d}.png")
                ti.size = len(img_data)
                tar.addfile(ti, io.BytesIO(img_data))

                tb = sample["txt"].encode("utf-8")
                tt = tarfile.TarInfo(name=f"sample_{start+idx:06d}.txt")
                tt.size = len(tb)
                tar.addfile(tt, io.BytesIO(tb))

        if (shard_id + 1) % max(1, num_shards // 10) == 0:
            print(f"  {shard_id+1}/{num_shards} shards")

    return shard_dir


class ShardedOCRDataset(Dataset):
    def __init__(self, shard_dir, processor, max_target_length=128, max_samples=None):
        self.processor         = processor
        self.tokenizer         = processor.tokenizer
        self.max_target_length = max_target_length
        self.samples           = []

        shard_files = sorted(f for f in os.listdir(shard_dir) if f.endswith(".tar"))
        for sf in shard_files:
            with tarfile.open(os.path.join(shard_dir, sf), "r") as tar:
                for m in tar.getmembers():
                    if m.name.endswith((".jpg", ".png")):
                        self.samples.append(
                            (os.path.join(shard_dir, sf), m.name.rsplit(".", 1)[0])
                        )
                        if max_samples and len(self.samples) >= max_samples:
                            break
            if max_samples and len(self.samples) >= max_samples:
                break

        if max_samples:
            self.samples = self.samples[:max_samples]

        print(f"Dataset: {len(self.samples):,} samples from {len(shard_files)} shards")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        shard_path, base = self.samples[idx]

        with tarfile.open(shard_path, "r") as tar:
            image = None
            for ext in (".jpg", ".png"):
                try:
                    image = Image.open(
                        tar.extractfile(tar.getmember(f"{base}{ext}"))
                    ).convert("RGB")
                    break
                except KeyError:
                    continue
            if image is None:
                raise FileNotFoundError(f"No image for {base}")

            text = tar.extractfile(
                tar.getmember(f"{base}.txt")
            ).read().decode("utf-8").strip()

        text = unicodedata.normalize("NFKC", text)

        # FIX 1: set src_lang AND tgt_lang per-sample so mBART prepends
        # the correct language token for this label's actual language.
        lang = LANG_CODE_BN if is_bangla(text) else LANG_CODE_EN
        self.tokenizer.src_lang = lang
        self.tokenizer.tgt_lang = lang

        pixel_values = self.processor(image, return_tensors="pt")["pixel_values"].squeeze()

        tokenized = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        )

        labels = tokenized.input_ids.squeeze()

        # FIX 2: strip the leading language token (bn_IN=250028 or en_XX=250004).
        # mBART tokenizer prepends src_lang as labels[0]. We remove it because
        # decoder_start_token_id (EOS, neutral seed) already handles decoder
        # initialisation. Keeping it caused a double BOS:
        #   [bn_IN(seed), bn_IN(from label), text...]  <- wrong
        # After fix:
        #   [</s>(seed),  text_token_1, text_token_2, ...]  <- correct
        # Shift left by 1, pad tail to preserve max_target_length.
        labels = torch.cat([
            labels[1:],
            torch.tensor([self.tokenizer.pad_token_id])
        ])

        labels[labels == self.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}


# ============================================================
# DATA SETUP
# 400k samples, 99/1 split -> ~4,000 val samples for stable CER.
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
    max_target_length=MAX_TARGET_LENGTH,
    max_samples=MAX_SAMPLES if TEST_MODE else None,
)

train_size = int(0.99 * len(data))
val_size   = len(data) - train_size
train_dataset, val_dataset = random_split(
    data, [train_size, val_size],
    generator=torch.Generator().manual_seed(42),
)
print(f"Train: {train_size:,}  Val: {val_size:,}")


# ============================================================
# METRICS
# ============================================================
def compute_metrics(pred):
    pred_ids  = pred.predictions
    label_ids = pred.label_ids

    pred_ids[pred_ids == -100]   = tokenizer.pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str  = [s.strip() for s in tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)]
    label_str = [s.strip() for s in tokenizer.batch_decode(label_ids, skip_special_tokens=True)]

    return {
        "cer": jiwer.cer(label_str, pred_str),
        "wer": jiwer.wer(label_str, pred_str),
    }


# ============================================================
# GENERATION CALLBACK
# Per-sample forced_bos_token_id and matching src/tgt_lang.
# ============================================================
class GenerationCallback(TrainerCallback):
    def __init__(
        self, processor, model, eval_images, eval_texts,
        tokenizer, output_dir="./outputs/runs"
    ):
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

        rep_count, total_len = 0, 0
        summary = f"## Generation Test — Step {step}

"
        self.model.eval()

        for i, (img_path, txt_path) in enumerate(zip(self.eval_images, self.eval_texts)):
            pil   = Image.open(img_path).convert("RGB")
            pv    = self.processor(images=pil, return_tensors="pt").pixel_values.to(self.model.device)
            truth = open(txt_path, encoding="utf-8").read().strip()

            # Per-sample language routing — matches training tokenization
            lang      = LANG_CODE_BN if is_bangla(truth) else LANG_CODE_EN
            lang_bos  = BN_TOKEN_ID  if is_bangla(truth) else EN_TOKEN_ID
            self.tokenizer.src_lang = lang
            self.tokenizer.tgt_lang = lang

            with torch.no_grad():
                outs = self.model.generate(
                    pv,
                    forced_bos_token_id=lang_bos,   # single language token, no duplicate
                    num_beams=4,
                    no_repeat_ngram_size=0,
                    max_length=MAX_TARGET_LENGTH,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                )

            # Encoder attention heatmap
            hm    = self.encoder_heatmap(outs.encoder_hidden_states[-1])
            hm_up = torch.nn.functional.interpolate(
                torch.tensor(hm).unsqueeze(0).unsqueeze(0),
                size=(pil.height, pil.width), mode="bilinear", align_corners=False
            )[0, 0].numpy()

            if self.writer:
                img_np   = np.array(pil).astype(np.float32) / 255.0
                h_norm   = (hm_up - hm_up.min()) / (hm_up.max() - hm_up.min() + 1e-8)
                import matplotlib.cm as cm
                hm_color = cm.get_cmap("viridis")(h_norm)[:, :, :3].astype(np.float32)
                overlay  = img_np * 0.5 + hm_color * 0.5
                self.writer.add_image(f"gen/overlay_{i+1}", np.transpose(overlay, (2,0,1)), step)
                self.writer.add_image(f"gen/image_{i+1}",   np.transpose(img_np,  (2,0,1)), step)

            gen_text = self.tokenizer.decode(outs.sequences[0], skip_special_tokens=True).strip()
            tokens   = outs.sequences[0].tolist()
            total_len += len(tokens)

            summary += f"### Sample {i+1} [{lang}]
"
            summary += f"**GT:**   {truth}

"
            summary += f"**Pred:** {gen_text}

"

            if len(tokens) > 3 and len(set(tokens[1:-1])) == 1:
                rep_count += 1
                summary += "WARNING: REPETITION DETECTED

"

        n    = max(len(self.eval_images), 1)
        avg  = total_len / n
        rate = rep_count / n * 100

        summary += f"**Repetitions:** {rep_count}/{n}  **Rate:** {rate:.1f}%  **Avg len:** {avg:.1f}
"
        if self.writer:
            self.writer.add_text("gen/samples", summary, step)
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
def extract_eval_samples_from_shards(shard_dir, num_samples=7):
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
    eval_images, eval_texts = extract_eval_samples_from_shards(shard_dir, num_samples=7)

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
        save_total_limit=3,
        predict_with_generate=True,
        generation_num_beams=4,
        generation_max_length=MAX_TARGET_LENGTH,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_steps=2000,
        max_grad_norm=1.0,
        load_best_model_at_end=True,
        eval_strategy="steps",
        eval_on_start=True,
        metric_for_best_model="eval_cer",
        greater_is_better=False,
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
            [EarlyStoppingCallback(early_stopping_patience=20)]
            + ([generation_callback] if generation_callback else [])
        ),
    )

    try:
        ckpts = (
            [f for f in os.listdir(training_args.output_dir) if f.startswith("checkpoint")]
            if os.path.exists(training_args.output_dir) else []
        )
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
                "decoder_model":          decoder_dir,
                "dataset_path":           DATA_DIR,
                "train_size":             train_size,
                "val_size":               val_size,
                "model_name":             hf_dir,
                "lang_code_bn":           LANG_CODE_BN,
                "lang_code_en":           LANG_CODE_EN,
                "learning_rate":          training_args.learning_rate,
                "lr_scheduler":           training_args.lr_scheduler_type,
                "weight_decay":           training_args.weight_decay,
                "warmup_steps":           training_args.warmup_steps,
                "grad_accum":             training_args.gradient_accumulation_steps,
                "max_target_length":      MAX_TARGET_LENGTH,
                "num_beams":              4,
                "encoder_layers_unfrozen":3,
                "decoder_start_token":    "eos (neutral)",
                "trainable_M":            round(
                    sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 2
                ),
            })