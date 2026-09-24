# ============================================================
# CONTROL EXPERIMENT
# IndicTrans2 + LoRA
# 5504 Sanskrit-English examples
# WITHOUT MORPHOLOGY
# ============================================================

import json
import random
from pathlib import Path

import numpy as np
import torch

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

DATA_FILE = (
    PROJECT_ROOT
    / "models"
    / "morph_lora"
    / "morphology_train_5504.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "morph_lora"
    / "baseline_5504"
)

MODEL_NAME = (
    "krpraveen/"
    "indictrans2-sanskrit-en-finetuned"
)


# ============================================================
# 2. SETTINGS
# ============================================================

SEED = 42

MAX_SOURCE_LENGTH = 256
MAX_TARGET_LENGTH = 256

BATCH_SIZE = 1
GRAD_ACCUMULATION = 4

LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01

NUM_EPOCHS = 2

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

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("INDICTRANS2 LoRA BASELINE — 5504")
print("CONTROL EXPERIMENT WITHOUT MORPHOLOGY")
print("=" * 70)

print()

print("Device:", DEVICE)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "GPU Memory:",
        round(
            torch.cuda.get_device_properties(0)
            .total_memory
            / 1024**3,
            2
        ),
        "GB"
    )

print()


# ============================================================
# 5. LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

if not DATA_FILE.exists():

    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_FILE}"
    )


with open(
    DATA_FILE,
    "r",
    encoding="utf-8"
) as f:

    all_records = json.load(f)


print(
    "Total records:",
    len(all_records)
)

print()


# ============================================================
# 6. VERIFY DATA
# ============================================================

if len(all_records) != 5504:

    print(
        "WARNING:"
        f" Expected 5504 records but found "
        f"{len(all_records)}"
    )


# ============================================================
# 7. REPRODUCIBLE TRAIN / VALIDATION SPLIT
# ============================================================

indices = list(
    range(len(all_records))
)

random.Random(SEED).shuffle(
    indices
)

val_size = int(
    len(indices) * 0.10
)

val_indices = indices[:val_size]

train_indices = indices[val_size:]


train_records = [
    all_records[i]
    for i in train_indices
]

validation_records = [
    all_records[i]
    for i in val_indices
]


print("=" * 70)
print("DATA SPLIT")
print("=" * 70)

print(
    "Training:",
    len(train_records)
)

print(
    "Validation:",
    len(validation_records)
)

print(
    "Seed:",
    SEED
)

print()


# ============================================================
# 8. IAST → DEVANAGARI
# ============================================================

def iast_to_devanagari(text):

    if not text:

        return ""

    return transliterate(
        text,
        sanscript.IAST,
        sanscript.DEVANAGARI
    )


# ============================================================
# 9. DATASET
# ============================================================

class BaselineDataset(Dataset):

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

        # ----------------------------------------------------
        # Sanskrit IAST → Devanagari
        # ----------------------------------------------------

        sanskrit_iast = record["sanskrit"]

        sanskrit_devanagari = (
            iast_to_devanagari(
                sanskrit_iast
            )
        )

        # ----------------------------------------------------
        # IMPORTANT
        #
        # No morphology is added here.
        #
        # This is the CONTROL experiment.
        # ----------------------------------------------------

        source = (
            "san_Deva "
            "eng_Latn "
            + sanskrit_devanagari
        )

        target = record["english"]

        # ----------------------------------------------------
        # SOURCE TOKENIZATION
        # ----------------------------------------------------

        model_inputs = self.tokenizer(
            source,
            max_length=self.max_source_length,
            truncation=True
        )

        # ----------------------------------------------------
        # TARGET TOKENIZATION
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
# 10. LOAD TOKENIZER
# ============================================================

print("=" * 70)
print("LOADING TOKENIZER")
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
# 11. CREATE DATASETS
# ============================================================

train_dataset = BaselineDataset(
    train_records,
    tokenizer,
    MAX_SOURCE_LENGTH,
    MAX_TARGET_LENGTH
)

validation_dataset = BaselineDataset(
    validation_records,
    tokenizer,
    MAX_SOURCE_LENGTH,
    MAX_TARGET_LENGTH
)


print("=" * 70)
print("DATASETS CREATED")
print("=" * 70)

print(
    "Train:",
    len(train_dataset)
)

print(
    "Validation:",
    len(validation_dataset)
)

print()


# ============================================================
# 12. SAMPLE CHECK
# ============================================================

print("=" * 70)
print("SAMPLE INPUT")
print("=" * 70)

sample_record = train_records[0]

sample_sanskrit = (
    iast_to_devanagari(
        sample_record["sanskrit"]
    )
)

sample_source = (
    "san_Deva "
    "eng_Latn "
    + sample_sanskrit
)

print()
print("IAST:")
print(sample_record["sanskrit"])

print()
print("Devanagari:")
print(sample_sanskrit)

print()
print("MODEL INPUT:")
print(sample_source)

print()
print("EXPECTED ENGLISH:")
print(sample_record["english"])

print()


# ============================================================
# 13. TEST TOKENIZATION
# ============================================================

sample_tokens = tokenizer(
    sample_source,
    max_length=MAX_SOURCE_LENGTH,
    truncation=True
)

print(
    "Tokenization successful."
)

print(
    "Token count:",
    len(sample_tokens["input_ids"])
)

print()


# ============================================================
# 14. LOAD BASE MODEL
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
# 15. LoRA
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
# 16. TRAINABLE PARAMETERS
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

print(
    "Epochs:",
    NUM_EPOCHS
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Gradient accumulation:",
    GRAD_ACCUMULATION
)

print(
    "Learning rate:",
    LEARNING_RATE
)

print(
    "LoRA rank:",
    LORA_R
)

print()


training_args = TrainingArguments(

    output_dir=str(
        OUTPUT_DIR
    ),

    overwrite_output_dir=True,

    num_train_epochs=NUM_EPOCHS,

    per_device_train_batch_size=(
        BATCH_SIZE
    ),

    per_device_eval_batch_size=1,

    gradient_accumulation_steps=(
        GRAD_ACCUMULATION
    ),

    learning_rate=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY,

    warmup_ratio=0.05,

    fp16=torch.cuda.is_available(),

    logging_steps=100,

    eval_strategy="epoch",

    save_strategy="epoch",

    save_total_limit=2,

    load_best_model_at_end=True,

    metric_for_best_model="eval_loss",

    greater_is_better=False,

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

    train_dataset=train_dataset,

    eval_dataset=validation_dataset,

    data_collator=data_collator
)


# ============================================================
# 20. TRAIN
# ============================================================

print("=" * 70)
print("STARTING BASELINE LoRA TRAINING")
print("=" * 70)

print()

train_result = trainer.train()


# ============================================================
# 21. FINAL EVALUATION LOSS
# ============================================================

print()

print("=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

eval_results = trainer.evaluate()

print()

for key, value in eval_results.items():

    print(
        f"{key}: {value}"
    )

print()


# ============================================================
# 22. SAVE MODEL
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


# ============================================================
# 23. SAVE TRAINING SUMMARY
# ============================================================

summary = {

    "experiment":
        "IndicTrans2 LoRA Baseline 5504",

    "morphology":
        False,

    "dataset_size":
        len(all_records),

    "train_size":
        len(train_records),

    "validation_size":
        len(validation_records),

    "seed":
        SEED,

    "epochs":
        NUM_EPOCHS,

    "batch_size":
        BATCH_SIZE,

    "gradient_accumulation":
        GRAD_ACCUMULATION,

    "learning_rate":
        LEARNING_RATE,

    "lora_r":
        LORA_R,

    "lora_alpha":
        LORA_ALPHA,

    "lora_dropout":
        LORA_DROPOUT,

    "training_loss":
        train_result.training_loss,

    "evaluation":
        eval_results
}


with open(
    OUTPUT_DIR / "training_summary.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# 24. GENERATION TEST
# ============================================================

print("=" * 70)
print("POST-TRAINING GENERATION TEST")
print("=" * 70)

model.eval()

test_record = validation_records[0]

test_iast = test_record["sanskrit"]

test_devanagari = (
    iast_to_devanagari(
        test_iast
    )
)

test_source = (
    "san_Deva "
    "eng_Latn "
    + test_devanagari
)

print()
print("Sanskrit IAST:")
print(test_iast)

print()
print("Sanskrit Devanagari:")
print(test_devanagari)

print()
print("MODEL INPUT:")
print(test_source)

print()
print("EXPECTED ENGLISH:")
print(test_record["english"])

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

        no_repeat_ngram_size=3,

        repetition_penalty=1.1,

        early_stopping=True

    )


generated_text = tokenizer.batch_decode(
    generated_ids,
    skip_special_tokens=True
)[0]


print("GENERATED ENGLISH:")
print(generated_text)

print()


# ============================================================
# 25. SAVE SAMPLE RESULT
# ============================================================

sample_result = {

    "sanskrit_iast":
        test_iast,

    "sanskrit_devanagari":
        test_devanagari,

    "expected":
        test_record["english"],

    "generated":
        generated_text
}


with open(
    OUTPUT_DIR / "sample_generation.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        sample_result,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# 26. FINISHED
# ============================================================

print("=" * 70)
print("BASELINE 5504 TRAINING COMPLETED")
print("=" * 70)

print()

print(
    "Model saved to:"
)

print(
    OUTPUT_DIR
)

print()

print(
    "This is the CONTROL model."
)

print(
    "It uses Sanskrit only and NO morphology."
)

print("=" * 70)