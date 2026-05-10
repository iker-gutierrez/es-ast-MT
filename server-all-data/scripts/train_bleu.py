import torch
import numpy as np
import evaluate

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
    TaskType
)

from config import *
from utils import preprocess

# =========================
# LOAD DATA
# =========================

dataset = load_from_disk(str(HF_DATASET_DIR))

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

tokenizer.src_lang = SRC_LANG

bleu = evaluate.load("sacrebleu")

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
# MODEL
# =========================

base_model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    low_cpu_mem_usage=True
)

forced_bos_token_id = tokenizer.convert_tokens_to_ids(TGT_LANG)

base_model.generation_config.forced_bos_token_id = forced_bos_token_id

# =========================
# GRADIENT CHECKPOINTING
# =========================

base_model.gradient_checkpointing_enable()

# =========================
# LORA
# =========================

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj"
    ],

    lora_dropout=0.05,

    bias="none",

    task_type=TaskType.SEQ_2_SEQ_LM
)

model = get_peft_model(base_model, lora_config)

model.print_trainable_parameters()

# =========================
# METRICS
# =========================

def compute_metrics(eval_preds):

    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    # Decode predictions
    decoded_preds = tokenizer.batch_decode(
        preds,
        skip_special_tokens=True
    )

    # Replace ignored index
    labels = np.where(
        labels != -100,
        labels,
        tokenizer.pad_token_id
    )

    # Decode labels
    decoded_labels = tokenizer.batch_decode(
        labels,
        skip_special_tokens=True
    )

    # Strip whitespace
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    # SacreBLEU expects references as list of lists
    decoded_labels = [[label] for label in decoded_labels]

    result = bleu.compute(
        predictions=decoded_preds,
        references=decoded_labels
    )

    return {
        "bleu": round(result["score"], 4)
    }

# =========================
# TRAINING
# =========================

training_args = Seq2SeqTrainingArguments(
    output_dir=str(MODEL_DIR),

    # =========================
    # BATCHING
    # =========================
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,

    gradient_accumulation_steps=2,

    # =========================
    # EPOCHS
    # =========================
    num_train_epochs=1.5,

    # =========================
    # OPTIMIZATION
    # =========================
    learning_rate=2e-4,

    lr_scheduler_type="cosine",

    warmup_ratio=0.05,

    weight_decay=0.01,

    # =========================
    # EVALUATION
    # =========================
    eval_strategy="steps",
    save_strategy="steps",

    eval_steps=2000,
    save_steps=2000,

    save_total_limit=2,

    load_best_model_at_end=True,

    metric_for_best_model="bleu",

    greater_is_better=True,

    # =========================
    # GENERATION
    # =========================
    predict_with_generate=True,

    generation_max_length=128,

    generation_num_beams=4,

    # =========================
    # PRECISION
    # =========================
    bf16=torch.cuda.is_bf16_supported(),

    fp16=not torch.cuda.is_bf16_supported(),

    # =========================
    # MEMORY
    # =========================
    gradient_checkpointing=True,

    # =========================
    # DATALOADER
    # =========================
    dataloader_num_workers=4,

    # =========================
    # LOGGING
    # =========================
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
# TRAIN
# =========================

trainer.train()

# =========================
# SAVE FINAL MODEL
# =========================

model.save_pretrained(str(MODEL_DIR))

tokenizer.save_pretrained(str(MODEL_DIR))