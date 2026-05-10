from datasets import load_dataset, Dataset, DatasetDict
from config import HF_DATASET_DIR

# =========================
# TRAIN SET
# =========================

aina = load_dataset("projecte-aina/ES-AST_Parallel_Corpus")

aina = aina.rename_columns({
    "es": "src",
    "ast": "tgt"
})

train_set = aina["train"].shuffle(seed=42)

# =========================
# FLORES
# =========================

def load_flores():

    es_dev = load_dataset(
        "openlanguagedata/flores_plus",
        "spa_Latn",
        split="dev"
    )

    ast_dev = load_dataset(
        "openlanguagedata/flores_plus",
        "ast_Latn",
        split="dev"
    )

    es_test = load_dataset(
        "openlanguagedata/flores_plus",
        "spa_Latn",
        split="devtest"
    )

    ast_test = load_dataset(
        "openlanguagedata/flores_plus",
        "ast_Latn",
        split="devtest"
    )

    dev_dataset = Dataset.from_dict({
        "src": es_dev["text"],
        "tgt": ast_dev["text"]
    })

    test_dataset = Dataset.from_dict({
        "src": es_test["text"],
        "tgt": ast_test["text"]
    })

    return dev_dataset, test_dataset


dev_set, test_set = load_flores()

dataset = DatasetDict({
    "train": train_set,
    "dev": dev_set,
    "test": test_set
})

HF_DATASET_DIR.mkdir(parents=True, exist_ok=True)

dataset.save_to_disk(str(HF_DATASET_DIR))

print(dataset)