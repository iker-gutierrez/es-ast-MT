import json
import torch
from tqdm.auto import tqdm

from pathlib import Path
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

import time

from config import *

import warnings

# =========================
# SILENCE WARNINGS
# =========================
warnings.filterwarnings("ignore")

# -----------------------
# paths
# -----------------------

dataset = load_from_disk(str(HF_DATASET_DIR))

test_src = list(dataset["test"]["src"])

# -----------------------
# tokenizer
# -----------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.src_lang = SRC_LANG

# -----------------------
# model
# -----------------------

# optimized for a ~42 GB GPU
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

model.to(DEVICE)
model.eval()

# optional speedup on recent GPUs
torch.backends.cuda.matmul.allow_tf32 = True

forced_bos_token_id = tokenizer.convert_tokens_to_ids(TGT_LANG)

# -----------------------
# generation
# -----------------------
start_time = time.time()

preds = []

# 42 GB VRAM should comfortably handle 32–64
BATCH_SIZE = 32

for i in tqdm(
    range(0, len(test_src), BATCH_SIZE),
    desc="Generating translations"
):

    batch = test_src[i:i + BATCH_SIZE]

    inputs = tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH
    )

    inputs = {
        k: v.to(DEVICE)
        for k, v in inputs.items()
    }

    with torch.inference_mode():

        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            max_length=None,
            forced_bos_token_id=forced_bos_token_id,
            num_beams=1,
            do_sample=False
        )

    batch_preds = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True
    )

    preds.extend(batch_preds)

    if DEVICE == "cuda":
        torch.cuda.empty_cache()


print("\n===== INFERENCE TIME STATS =====")
end_time = time.time()

total_time = end_time - start_time
num_sentences = len(test_src)

sent_per_sec = num_sentences / total_time
avg_sent_time = total_time / num_sentences

print(f"Total time: {total_time:.2f} seconds")
print(f"Sentences: {num_sentences}")
print(f"Sentences/sec: {sent_per_sec:.2f}")
print(f"Avg sentence latency: {avg_sent_time:.4f} sec")
# -----------------------
# save predictions
# -----------------------

BASELINE_PRED_DIR.mkdir(parents=True, exist_ok=True)

with open(
    BASELINE_PRED_DIR / "predictions.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(preds, f, ensure_ascii=False, indent=2)

with open(
    BASELINE_PRED_DIR / "references.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        list(dataset["test"]["tgt"]),
        f,
        ensure_ascii=False,
        indent=2
    )

time_stats = {
    "total_time_seconds": total_time,
    "num_sentences": num_sentences,
    "sentences_per_second": sent_per_sec,
    "avg_sentence_latency_seconds": avg_sent_time,
    "batch_size": BATCH_SIZE,
    "model": MODEL_NAME
}

with open(
    BASELINE_PRED_DIR / "inference_metrics.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(time_stats, f, indent=2)