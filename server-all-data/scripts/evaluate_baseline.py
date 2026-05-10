import json
from pathlib import Path

import sacrebleu
import evaluate

# =========================
# PATHS
# =========================

PRED_PATH = Path("./workspace/predictions_baseline/predictions.json")
REF_PATH = Path("./workspace/predictions_baseline/references.json")
OUTPUT_PATH = Path("./workspace/predictions_baseline/metrics_results.json")

# =========================
# LOAD DATA
# =========================

with open(PRED_PATH, "r", encoding="utf-8") as f:
    preds = json.load(f)

with open(REF_PATH, "r", encoding="utf-8") as f:
    refs = json.load(f)

# =========================
# BLEU
# =========================

bleu = sacrebleu.corpus_bleu(preds, [refs])
bleu_score = bleu.score

print(f"\nBLEU: {bleu_score:.2f}")

# =========================
# chrF
# =========================

chrf = sacrebleu.corpus_chrf(preds, [refs], word_order=0)
chrf_score = chrf.score
print(f"chrF: {chrf_score:.2f}")

# =========================
# chrF++
# =========================

chrfpp = sacrebleu.corpus_chrf(preds, [refs], word_order=2)
chrfpp_score = chrfpp.score
print(f"chrF++: {chrfpp_score:.2f}")

# =========================
# COMET
# =========================

print("\nLoading COMET...")

comet = evaluate.load("comet")

sources = [""] * len(preds)  # replace with real sources if available

comet_result = comet.compute(
    sources=sources,
    predictions=preds,
    references=refs
)

comet_score = comet_result["mean_score"] * 100

print(f"COMET: {comet_score:.4f}")

# =========================
# BLEURT
# =========================

print("\nLoading BLEURT...")

bleurt = evaluate.load("bleurt", "BLEURT-20")

bleurt_result = bleurt.compute(
    predictions=preds,
    references=refs
)

bleurt_score = sum(bleurt_result["scores"]) / len(bleurt_result["scores"]) * 100

print(f"BLEURT: {bleurt_score:.4f}")

# =========================
# SAVE RESULTS
# =========================

results = {
    "BLEU": round(bleu_score, 4),
    "chrF": round(chrf_score, 4),
    "chrF++": round(chrfpp_score, 4),
    "COMET": round(comet_score, 4),
    "BLEURT": round(bleurt_score, 4),
}

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"\nSaved results to: {OUTPUT_PATH}")