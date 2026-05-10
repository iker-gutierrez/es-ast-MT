import json
import warnings
import torch
from tqdm.auto import tqdm
import time
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    logging
)
from peft import PeftModel
from config import *

# =========================
# SILENCE WARNINGS
# =========================
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

# =========================
# LOAD DATASET
# =========================
dataset = load_from_disk(str(HF_DATASET_DIR))
test_src = list(dataset["test"]["src"])

# =========================
# TOKENIZER
# =========================
tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_DIR)
)
tokenizer.src_lang = SRC_LANG

# =========================
# MODEL
# =========================
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
model = PeftModel.from_pretrained(
    base_model,
    str(MODEL_DIR)
)
model.to(DEVICE)
model.eval()

# optional speedup on recent GPUs
torch.backends.cuda.matmul.allow_tf32 = True

# remove transformers warning
model.generation_config.max_length = None

forced_bos_token_id = tokenizer.convert_tokens_to_ids(
    TGT_LANG
)

# =========================
# GENERATION
# =========================
start_time = time.time()
preds = []

# safe for a 42GB GPU
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
            max_new_tokens=128,
            num_beams=5,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
            early_stopping=True,
            forced_bos_token_id=forced_bos_token_id
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

# =========================
# SAVE OUTPUTS
# =========================
FT_PRED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    FT_PRED_DIR / "predictions.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        preds,
        f,
        ensure_ascii=False,
        indent=2
    )

with open(
    FT_PRED_DIR / "references.json",
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
    FT_PRED_DIR / "inference_metrics.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(time_stats, f, indent=2)
