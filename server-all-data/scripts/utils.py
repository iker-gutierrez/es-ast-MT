from transformers import AutoTokenizer

from config import (
    MODEL_NAME,
    SRC_LANG,
    TGT_LANG,
    MAX_LENGTH
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

tokenizer.src_lang = SRC_LANG

def preprocess(examples):

    return tokenizer(
        examples["src"],
        text_target=examples["tgt"],
        src_lang=SRC_LANG,
        tgt_lang=TGT_LANG,
        max_length=MAX_LENGTH,
        truncation=True
    )
