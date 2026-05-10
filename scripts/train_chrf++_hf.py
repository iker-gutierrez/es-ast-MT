import torch
import numpy as np
import evaluate
import sacrebleu
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)
from config import *
from utils import preprocess

# =========================
# LOAD DATA
# =========================
dataset = load_from_disk(str(HF_DATASET_DIR))
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.src_lang = SRC_LANG
chrf = evaluate.load("chrf")

# =========================
# TOKENIZATION
# =========================
tokenized_datasets = {
    split: ds.map(
        preprocess,
        batched=True,
        num_proc=4,
        remove_columns=ds.column_names
    )
    for split, ds in dataset.items()
}
tok_train = tokenized_datasets["train"]
tok_dev = tokenized_datasets["dev"]

# =========================
# MODEL SETUP
# =========================
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    low_cpu_mem_usage=True
)

# Generation config safety
forced_bos_token_id = tokenizer.convert_tokens_to_ids(TGT_LANG)
model.generation_config.forced_bos_token_id = forced_bos_token_id

# Setup LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)
model = get_peft_model(model, lora_config)
model.config.use_cache = False  # Required for gradient checkpointing

model.print_trainable_parameters()

# =========================
# METRICS
# =========================
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = [p.strip() for p in decoded_preds]
    decoded_labels = [[l.strip()] for l in decoded_labels]
    result = chrf.compute(predictions=decoded_preds, references=decoded_labels, word_order=2)
    return {"chrf++": round(result["score"], 4)}

# =========================
# TRAINING ARGS
# =========================
training_args = Seq2SeqTrainingArguments(
    output_dir=str(MODEL_DIR),
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=1.5,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=2000,
    save_steps=2000,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="chrf++",
    greater_is_better=True,
    predict_with_generate=True,
    generation_max_length=128,
    generation_num_beams=4,
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},  # avoids the requires_grad hook issue
    dataloader_num_workers=4,
    logging_steps=200,
    report_to="none",
    seed=42
)

# =========================
# DATA COLLATOR
# =========================
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    padding=True
)

# =========================
# TRAINER
# =========================
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tok_train,
    eval_dataset=tok_dev,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# =========================
# TRAIN & SAVE
# =========================
trainer.train()
model.save_pretrained(str(MODEL_DIR))
tokenizer.save_pretrained(str(MODEL_DIR))
