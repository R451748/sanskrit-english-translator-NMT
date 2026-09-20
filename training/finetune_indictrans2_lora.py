import os
import torch
import pandas as pd

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)

from peft import LoraConfig, get_peft_model, TaskType
from indic_transliteration import sanscript
from IndicTransToolkit import IndicProcessor


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = "/mnt/c/Users/rohan/MTech_Sanskrit_NMT"

MODEL_NAME = "krpraveen/indictrans2-sanskrit-en-finetuned"

TRAIN_FILE = os.path.join(
    PROJECT_ROOT,
    "dataset/final/train.csv"
)

VAL_FILE = os.path.join(
    PROJECT_ROOT,
    "dataset/final/validation.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "models/indictrans2_lora_25k"
)

TRAIN_SAMPLES = 25000
VAL_SAMPLES = 2500

SEED = 42


# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DTYPE = (
    torch.float16
    if DEVICE == "cuda"
    else torch.float32
)

print("=" * 60)
print("IndicTrans2 LoRA Fine-Tuning - 25K")
print("=" * 60)

print("Device:", DEVICE)

if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)

# Reproducible random subsets
train_df = train_df.sample(
    n=TRAIN_SAMPLES,
    random_state=SEED
).reset_index(drop=True)

val_df = val_df.sample(
    n=VAL_SAMPLES,
    random_state=SEED
).reset_index(drop=True)

print("Training samples:", len(train_df))
print("Validation samples:", len(val_df))


# ============================================================
# IAST -> DEVANAGARI
# ============================================================

print("\nConverting IAST -> Devanagari...")

train_src = [
    sanscript.transliterate(
        str(x),
        sanscript.IAST,
        sanscript.DEVANAGARI
    )
    for x in train_df["sanskrit"]
]

train_tgt = [
    str(x)
    for x in train_df["english"]
]

val_src = [
    sanscript.transliterate(
        str(x),
        sanscript.IAST,
        sanscript.DEVANAGARI
    )
    for x in val_df["sanskrit"]
]

val_tgt = [
    str(x)
    for x in val_df["english"]
]


# ============================================================
# INDIC PROCESSOR
# ============================================================

print("\nInitializing IndicProcessor...")

ip = IndicProcessor(inference=True)

train_src = ip.preprocess_batch(
    train_src,
    src_lang="san_Deva",
    tgt_lang="eng_Latn"
)

val_src = ip.preprocess_batch(
    val_src,
    src_lang="san_Deva",
    tgt_lang="eng_Latn"
)


# ============================================================
# DATASETS
# ============================================================

train_dataset = Dataset.from_dict({
    "source": train_src,
    "target": train_tgt
})

val_dataset = Dataset.from_dict({
    "source": val_src,
    "target": val_tgt
})


# ============================================================
# TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_function(examples):

    model_inputs = tokenizer(
        examples["source"],
        max_length=256,
        truncation=True
    )

    labels = tokenizer(
        text_target=examples["target"],
        max_length=256,
        truncation=True
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


print("\nTokenizing training data...")

tokenized_train = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=train_dataset.column_names
)

print("\nTokenizing validation data...")

tokenized_val = val_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=val_dataset.column_names
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading IndicTrans2 model...")

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=DTYPE
)


# ============================================================
# LoRA
# ============================================================

print("\nConfiguring LoRA...")

lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,

    r=8,

    lora_alpha=16,

    lora_dropout=0.05,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj"
    ],

    bias="none"
)

model = get_peft_model(
    model,
    lora_config
)

model.print_trainable_parameters()


# ============================================================
# DATA COLLATOR
# ============================================================

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)


# ============================================================
# TRAINING
# ============================================================

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    overwrite_output_dir=True,

    num_train_epochs=2,

    per_device_train_batch_size=1,

    per_device_eval_batch_size=1,

    gradient_accumulation_steps=4,

    learning_rate=2e-4,

    weight_decay=0.01,

    warmup_ratio=0.05,

    fp16=(DEVICE == "cuda"),

    logging_steps=100,

    eval_strategy="epoch",

    save_strategy="epoch",

    save_total_limit=2,

    load_best_model_at_end=True,

    metric_for_best_model="eval_loss",

    greater_is_better=False,

    report_to="none",

    seed=SEED
)


# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=tokenized_train,

    eval_dataset=tokenized_val,

    tokenizer=tokenizer,

    data_collator=data_collator
)


# ============================================================
# START TRAINING
# ============================================================

print("\n")
print("=" * 60)
print("STARTING 25K LoRA TRAINING")
print("=" * 60)

trainer.train()


# ============================================================
# SAVE
# ============================================================

print("\nSaving LoRA adapter...")

trainer.save_model(OUTPUT_DIR)

tokenizer.save_pretrained(OUTPUT_DIR)

print("\nTraining complete.")

print("Saved to:")
print(OUTPUT_DIR)