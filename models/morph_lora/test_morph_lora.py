# ============================================================
# Morphology-Conditioned IndicTrans2 LoRA
# SANITY TEST
# ============================================================

import os
import json
import random
import numpy as np
import torch

from pathlib import Path
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

from peft import (
    LoraConfig,
    get_peft_model,
)

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(
    "/mnt/c/Users/rohan/MTech_Sanskrit_NMT"
)

MORPH_FILE = (
    PROJECT_ROOT
    / "models"
    / "morph_lora"
    / "train.json"
)

# OUTPUT_DIR = (
#     PROJECT_ROOT
#     / "models"
#     / "morph_lora"
#     / "sanity_test"
# )
OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "morph_lora"
    / "test_200_steps"
)

MODEL_NAME = "krpraveen/indictrans2-sanskrit-en-finetuned"


# ============================================================
# 2. SETTINGS
# ============================================================

SEED = 42

MAX_SOURCE_LENGTH = 256
MAX_TARGET_LENGTH = 256

# SANITY_STEPS = 20
SANITY_STEPS = 200
BATCH_SIZE = 1
GRAD_ACCUMULATION = 4

LEARNING_RATE = 2e-4

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05


# ============================================================
# 3. SEED
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# 4. DEVICE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("MORPHOLOGY-CONDITIONED INDIC TRANS2 LoRA SANITY TEST")
print("=" * 70)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    print(
        f"GPU Memory: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )

print()


# ============================================================
# 5. LOAD MORPHOLOGY DATA
# ============================================================

print("=" * 70)
print("LOADING MORPHOLOGY DATA")
print("=" * 70)

if not MORPH_FILE.exists():

    raise FileNotFoundError(
        f"Morphology dataset not found:\n{MORPH_FILE}"
    )


with open(
    MORPH_FILE,
    "r",
    encoding="utf-8"
) as f:

    records = json.load(f)


print(f"Records loaded: {len(records)}")

if len(records) == 0:

    raise ValueError(
        "Morphology dataset is empty."
    )

print()


# ============================================================
# 6. CHECK DATA
# ============================================================

print("=" * 70)
print("DATA SAMPLE")
print("=" * 70)

sample_record = records[0]

print(
    "Index:",
    sample_record.get("index")
)

print(
    "Sanskrit IAST:",
    sample_record.get("sanskrit")
)

print(
    "Morphology:",
    sample_record.get("morphology_tags")
)

print(
    "English:",
    sample_record.get("english")
)

print()


# ============================================================
# 7. IAST → DEVANAGARI
# ============================================================

def iast_to_devanagari(text):

    if not text:
        return ""

    try:

        return transliterate(
            text,
            sanscript.IAST,
            sanscript.DEVANAGARI
        )

    except Exception as e:

        print(
            "Transliteration warning:",
            e
        )

        return text


# ============================================================
# 8. BUILD MORPHOLOGY INPUT
# ============================================================

def build_source(record):

    sanskrit_iast = record["sanskrit"]

    sanskrit_devanagari = (
        iast_to_devanagari(
            sanskrit_iast
        )
    )

    morphology_tags = record.get(
        "morphology_tags",
        []
    )

    morphology = " ".join(
        morphology_tags
    )

    # IMPORTANT:
    #
    # IndicTrans2 tokenizer expects:
    #
    # source language
    # target language
    # actual source sentence
    #
    # Therefore:
    #
    # san_Deva eng_Latn Sanskrit MORPHOLOGY tags
    #
    source = (
        "san_Deva "
        "eng_Latn "
        + sanskrit_devanagari
        + " MORPHOLOGY "
        + morphology
    )

    return (
        sanskrit_iast,
        sanskrit_devanagari,
        morphology,
        source
    )


# ============================================================
# 9. LOAD TOKENIZER
# ============================================================

print("=" * 70)
print("LOADING INDIC TRANS2 TOKENIZER")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

print(
    "Tokenizer:",
    tokenizer.__class__.__name__
)

print()


# ============================================================
# 10. SAMPLE TOKENIZATION TEST
# ============================================================

print("=" * 70)
print("TOKENIZATION TEST")
print("=" * 70)

(
    sample_iast,
    sample_devanagari,
    sample_morphology,
    sample_source
) = build_source(
    sample_record
)

print()
print("IAST:")
print(sample_iast)

print()
print("Devanagari:")
print(sample_devanagari)

print()
print("Morphology:")
print(sample_morphology)

print()
print("FINAL MODEL INPUT:")
print(sample_source)

print()


try:

    sample_tokens = tokenizer(
        sample_source,
        max_length=MAX_SOURCE_LENGTH,
        truncation=True
    )

    print(
        "Tokenization: SUCCESS"
    )

    print(
        "Number of tokens:",
        len(sample_tokens["input_ids"])
    )

except Exception as e:

    print()
    print(
        "TOKENIZATION FAILED"
    )

    print(
        repr(e)
    )

    raise


print()


# ============================================================
# 11. DATASET
# ============================================================

class MorphologyDataset(Dataset):

    def __init__(
        self,
        records,
        tokenizer,
        max_source_length=256,
        max_target_length=256
    ):

        self.records = records

        self.tokenizer = tokenizer

        self.max_source_length = (
            max_source_length
        )

        self.max_target_length = (
            max_target_length
        )

    def __len__(self):

        return len(self.records)

    def __getitem__(self, idx):

        record = self.records[idx]

        (
            _,
            _,
            _,
            source
        ) = build_source(record)

        target = record["english"]

        # ----------------------------------------------------
        # SOURCE
        # ----------------------------------------------------

        model_inputs = self.tokenizer(
            source,
            max_length=self.max_source_length,
            truncation=True
        )

        # ----------------------------------------------------
        # TARGET
        # ----------------------------------------------------

        labels = self.tokenizer(
            text_target=target,
            max_length=self.max_target_length,
            truncation=True
        )

        model_inputs["labels"] = (
            labels["input_ids"]
        )

        return model_inputs


# ============================================================
# 12. CREATE DATASET
# ============================================================

print("=" * 70)
print("CREATING DATASET")
print("=" * 70)

# Use a small number of records for sanity testing.
# sanity_records = records[:64]
sanity_records = records[:256]
dataset = MorphologyDataset(
    sanity_records,
    tokenizer,
    MAX_SOURCE_LENGTH,
    MAX_TARGET_LENGTH
)

print(
    f"Sanity dataset size: {len(dataset)}"
)

print()


# ============================================================
# 13. CHECK DATASET SAMPLE
# ============================================================

print("=" * 70)
print("DATASET SAMPLE")
print("=" * 70)

item = dataset[0]

print(
    "Input IDs:",
    len(item["input_ids"])
)

print(
    "Attention mask:",
    len(item["attention_mask"])
)

print(
    "Labels:",
    len(item["labels"])
)

print()


# ============================================================
# 14. LOAD MODEL
# ============================================================

print("=" * 70)
print("LOADING BASE MODEL")
print("=" * 70)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=(
        torch.float16
        if torch.cuda.is_available()
        else torch.float32
    )
)

print(
    "Base model loaded."
)

print()


# ============================================================
# 15. LoRA CONFIGURATION
# ============================================================

print("=" * 70)
print("CONFIGURING LoRA")
print("=" * 70)

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj"
    ],

    bias="none",

    task_type="SEQ_2_SEQ_LM"
)


model = get_peft_model(
    model,
    lora_config
)


# ============================================================
# 16. PRINT TRAINABLE PARAMETERS
# ============================================================

print()

model.print_trainable_parameters()

print()


# ============================================================
# 17. DATA COLLATOR
# ============================================================

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)


# ============================================================
# 18. TRAINING ARGUMENTS
# ============================================================

print("=" * 70)
print("TRAINING CONFIGURATION")
print("=" * 70)

training_args = TrainingArguments(

    output_dir=str(
        OUTPUT_DIR
    ),

    overwrite_output_dir=True,

    max_steps=SANITY_STEPS,

    per_device_train_batch_size=BATCH_SIZE,

    gradient_accumulation_steps=(
        GRAD_ACCUMULATION
    ),

    learning_rate=LEARNING_RATE,

    fp16=torch.cuda.is_available(),

    logging_steps=1,

    save_strategy="no",

    eval_strategy="no",

    report_to="none",

    dataloader_num_workers=0,

    remove_unused_columns=False,

    seed=SEED,

    optim="adamw_torch"
)


# ============================================================
# 19. TRAINER
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=dataset,

    data_collator=data_collator,

)


# ============================================================
# 20. TRAIN
# ============================================================

print("=" * 70)
print("STARTING MORPHOLOGY LoRA SANITY TRAINING")
print("=" * 70)

print(
    f"Steps: {SANITY_STEPS}"
)

print(
    f"Batch size: {BATCH_SIZE}"
)

print(
    f"Gradient accumulation: "
    f"{GRAD_ACCUMULATION}"
)

print(
    f"Learning rate: {LEARNING_RATE}"
)

print()

train_result = trainer.train()


# ============================================================
# 21. TRAINING RESULT
# ============================================================

print()
print("=" * 70)
print("SANITY TRAINING COMPLETED")
print("=" * 70)

print(
    "Training loss:",
    train_result.training_loss
)

print()


# ============================================================
# 22. SAVE SANITY MODEL
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

trainer.save_model(
    str(OUTPUT_DIR)
)

tokenizer.save_pretrained(
    str(OUTPUT_DIR)
)

print(
    "Sanity model saved to:"
)

print(
    OUTPUT_DIR
)

print()


# ============================================================
# 23. GENERATION TEST
# ============================================================

print("=" * 70)
print("POST-TRAINING GENERATION TEST")
print("=" * 70)

model.eval()

generation_record = records[0]

(
    test_iast,
    test_devanagari,
    test_morphology,
    test_source
) = build_source(
    generation_record
)

print()
print("Sanskrit IAST:")
print(test_iast)

print()
print("Sanskrit Devanagari:")
print(test_devanagari)

print()
print("Morphology:")
print(test_morphology)

print()
print("Model input:")
print(test_source)

print()


inputs = tokenizer(
    test_source,
    return_tensors="pt",
    max_length=MAX_SOURCE_LENGTH,
    truncation=True
)


inputs = {
    key: value.to(model.device)
    for key, value in inputs.items()
}


with torch.no_grad():

    generated_ids = model.generate(

        **inputs,

        max_length=128,

        num_beams=2,

        early_stopping=True

    )


generated_text = tokenizer.batch_decode(
    generated_ids,
    skip_special_tokens=True
)[0]


print("EXPECTED ENGLISH:")
print(generation_record["english"])

print()

print("GENERATED ENGLISH:")
print(generated_text)

print()


# ============================================================
# 24. FINAL STATUS
# ============================================================

print("=" * 70)
print("SANITY TEST FINISHED")
print("=" * 70)

print()
print("If you reached this point:")
print("1. Tokenizer accepted the morphology input.")
print("2. IndicTrans2 model loaded.")
print("3. LoRA was attached.")
print("4. Training completed.")
print("5. Generation was executed.")
print()
print("Next step: compare morphology-LoRA against")
print("the existing 25K standard LoRA model.")
print("=" * 70)