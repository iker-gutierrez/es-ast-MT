import json

with open("./workspace/predictions_ft/predictions.json") as f:
    preds = json.load(f)

with open("./workspace/predictions_ft/references.json") as f:
    refs = json.load(f)

print(f"Num predictions: {len(preds)}")
print(f"Num references:  {len(refs)}")
print()
for i in range(5):
    print(f"[{i}] PRED: {preds[i]}")
    print(f"[{i}] REF:  {refs[i]}")
    print()