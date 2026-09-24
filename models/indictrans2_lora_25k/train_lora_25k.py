# import os
# import random
# import numpy as np
# import pandas as pd
# import torch

# from datasets import Dataset
# from transformers import (
#     AutoTokenizer,
#     AutoModelForSeq2SeqLM,
#     DataCollatorForSeq2Seq,
#     Seq2SeqTrainingArguments,
#     Seq2SeqTrainer,
# )

# from peft import LoraConfig, get_peft_model, TaskType
# from indic_transliteration import sanscript
# from IndicTransToolkit import IndicProcessor


# # ============================================================
# # CONFIGURATION
# # ============================================================

# MODEL_NAME = "krpraveen/indictrans2-sanskrit-en-finetuned"

# PROJECT_ROOT = "/mnt/c/Users/rohan/MTech_Sanskrit_NMT"

# TRAIN_FILE = os.path.join(
#     PROJECT_ROOT,
#     "dataset",
#     "final",
#     "lora_train_25k.csv"
# )

# VAL_FILE = os.path.join(
#     PROJECT_ROOT,
#     "dataset",
#     "final",
#     "validation.csv"
# )

# OUTPUT_DIR = os.path.join(
#     PROJECT_ROOT,
#     "models",
#     "indictrans2_lora_25k_full"
# )

# TRAIN_SAMPLES = 25000
# VAL_SAMPLES = 2500

# SEED = 42


# # ============================================================
# # REPRODUCIBILITY
# # ============================================================

# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)

# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(SEED)


# # ============================================================
# # DEVICE
# # ============================================================

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# print("=" * 70)
# print("INDICTRANS2 LoRA FINE-TUNING")
# print("=" * 70)

# print("Device:", DEVICE)

# if torch.cuda.is_available():
#     print("GPU:", torch.cuda.get_device_name(0))
#     print(
#         "GPU Memory:",
#         round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
#         "GB"
#     )

# print("Model:", MODEL_NAME)
# print("Training samples:", TRAIN_SAMPLES)
# print("Validation samples:", VAL_SAMPLES)
# print("=" * 70)


# # ============================================================
# # LOAD DATA
# # ============================================================

# print("\nLoading training data...")

# train_df = pd.read_csv(TRAIN_FILE)
# val_df = pd.read_csv(VAL_FILE)

# train_df = train_df.dropna(subset=["sanskrit", "english"])
# val_df = val_df.dropna(subset=["sanskrit", "english"])

# train_df = train_df.sample(
#     n=min(TRAIN_SAMPLES, len(train_df)),
#     random_state=SEED
# ).reset_index(drop=True)

# val_df = val_df.sample(
#     n=min(VAL_SAMPLES, len(val_df)),
#     random_state=SEED
# ).reset_index(drop=True)

# print("Train rows:", len(train_df))
# print("Validation rows:", len(val_df))


# # ============================================================
# # IAST → DEVANAGARI
# # ============================================================

# print("\nConverting Sanskrit IAST → Devanagari...")

# def convert_to_devanagari(text):
#     try:
#         return sanscript.transliterate(
#             str(text),
#             sanscript.IAST,
#             sanscript.DEVANAGARI
#         )
#     except Exception:
#         return str(text)


# train_sanskrit = [
#     convert_to_devanagari(x)
#     for x in train_df["sanskrit"]
# ]

# val_sanskrit = [
#     convert_to_devanagari(x)
#     for x in val_df["sanskrit"]
# ]


# # ============================================================
# # CREATE HUGGING FACE DATASETS
# # ============================================================

# train_dataset = Dataset.from_dict({
#     "source": train_sanskrit,
#     "target": train_df["english"].astype(str).tolist()
# })

# val_dataset = Dataset.from_dict({
#     "source": val_sanskrit,
#     "target": val_df["english"].astype(str).tolist()
# })


# # ============================================================
# # LOAD TOKENIZER
# # ============================================================

# print("\nLoading tokenizer...")

# tokenizer = AutoTokenizer.from_pretrained(
#     MODEL_NAME,
#     trust_remote_code=True
# )

# print("Tokenizer loaded.")


# # ============================================================
# # LOAD MODEL
# # ============================================================

# print("\nLoading IndicTrans2 model...")

# DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# model = AutoModelForSeq2SeqLM.from_pretrained(
#     MODEL_NAME,
#     trust_remote_code=True,
#     torch_dtype=DTYPE
# )

# print("Base model loaded.")


# # ============================================================
# # LOADING INDIC PROCESSOR
# # ============================================================

# processor = IndicProcessor(
#     inference=True
# )


# # ============================================================
# # TOKENIZATION
# # ============================================================

# MAX_SOURCE_LENGTH = 256
# MAX_TARGET_LENGTH = 256


# def preprocess_function(examples):

#     processed_sources = []

#     for text in examples["source"]:

#         processed = processor.preprocess_batch(
#             [text],
#             src_lang="san_Deva",
#             tgt_lang="eng_Latn"
#         )

#         processed_sources.append(processed[0])

#     model_inputs = tokenizer(
#         processed_sources,
#         max_length=MAX_SOURCE_LENGTH,
#         truncation=True
#     )

#     labels = tokenizer(
#         text_target=examples["target"],
#         max_length=MAX_TARGET_LENGTH,
#         truncation=True
#     )

#     model_inputs["labels"] = labels["input_ids"]

#     return model_inputs


# print("\nTokenizing training dataset...")

# tokenized_train = train_dataset.map(
#     preprocess_function,
#     batched=True,
#     remove_columns=train_dataset.column_names,
#     desc="Tokenizing train"
# )

# print("\nTokenizing validation dataset...")

# tokenized_val = val_dataset.map(
#     preprocess_function,
#     batched=True,
#     remove_columns=val_dataset.column_names,
#     desc="Tokenizing validation"
# )

# print("Tokenization complete.")


# # ============================================================
# # LoRA CONFIGURATION
# # ============================================================

# print("\nApplying LoRA...")

# lora_config = LoraConfig(
#     task_type=TaskType.SEQ_2_SEQ_LM,

#     r=8,

#     lora_alpha=16,

#     lora_dropout=0.05,

#     target_modules=[
#         "q_proj",
#         "k_proj",
#         "v_proj",
#         "out_proj"
#     ],

#     bias="none"
# )

# model = get_peft_model(
#     model,
#     lora_config
# )

# model.print_trainable_parameters()


# # ============================================================
# # DATA COLLATOR
# # ============================================================

# data_collator = DataCollatorForSeq2Seq(
#     tokenizer=tokenizer,
#     model=model,
#     padding=True
# )


# # ============================================================
# # TRAINING ARGUMENTS
# # ============================================================

# training_args = Seq2SeqTrainingArguments(

#     output_dir=OUTPUT_DIR,

#     # --------------------------------------------------------
#     # TRAINING
#     # --------------------------------------------------------

#     num_train_epochs=2,

#     per_device_train_batch_size=1,

#     per_device_eval_batch_size=1,

#     gradient_accumulation_steps=4,

#     learning_rate=2e-4,

#     weight_decay=0.01,

#     warmup_ratio=0.05,

#     max_grad_norm=1.0,

#     # --------------------------------------------------------
#     # PRECISION
#     # --------------------------------------------------------

#     fp16=torch.cuda.is_available(),

#     # --------------------------------------------------------
#     # EVALUATION
#     # --------------------------------------------------------

#     eval_strategy="epoch",

#     # --------------------------------------------------------
#     # SAVING
#     # --------------------------------------------------------

#     save_strategy="epoch",

#     save_total_limit=2,

#     load_best_model_at_end=True,

#     metric_for_best_model="eval_loss",

#     greater_is_better=False,

#     # --------------------------------------------------------
#     # LOGGING
#     # --------------------------------------------------------

#     logging_strategy="steps",

#     logging_steps=100,

#     # --------------------------------------------------------
#     # PERFORMANCE
#     # --------------------------------------------------------

#     dataloader_num_workers=0,

#     # --------------------------------------------------------
#     # OTHER
#     # --------------------------------------------------------

#     report_to="none",

#     seed=SEED,

#     remove_unused_columns=True
# )


# # ============================================================
# # TRAINER
# # ============================================================

# trainer = Seq2SeqTrainer(

#     model=model,

#     args=training_args,

#     train_dataset=tokenized_train,

#     eval_dataset=tokenized_val,

#     processing_class=tokenizer,

#     data_collator=data_collator
# )


# # ============================================================
# # TRAIN
# # ============================================================

# print("\n")
# print("=" * 70)
# print("STARTING FULL 25K LoRA TRAINING")
# print("=" * 70)

# trainer.train()


# # ============================================================
# # FINAL EVALUATION LOSS
# # ============================================================

# print("\n")
# print("=" * 70)
# print("FINAL VALIDATION")
# print("=" * 70)

# results = trainer.evaluate()

# print(results)


# # ============================================================
# # SAVE LoRA ADAPTER
# # ============================================================

# print("\nSaving final LoRA adapter...")

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# model.save_pretrained(OUTPUT_DIR)

# tokenizer.save_pretrained(OUTPUT_DIR)

# print("\n" + "=" * 70)
# print("TRAINING COMPLETE")
# print("=" * 70)

# print("LoRA adapter saved to:")

# print(OUTPUT_DIR)

# print("=" * 70)

import os
import random
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

from peft import LoraConfig, get_peft_model, TaskType
from indic_transliteration import sanscript
from IndicTransToolkit import IndicProcessor


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "krpraveen/indictrans2-sanskrit-en-finetuned"

PROJECT_ROOT = "/mnt/c/Users/rohan/MTech_Sanskrit_NMT"

TRAIN_FILE = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "final",
    "lora_train_25k.csv"
)

VAL_FILE = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "final",
    "validation.csv"
)

# IMPORTANT:
# Keep the safety-test output and full-training output separate.
OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
    "indictrans2_lora_25k_full"
)

TRAIN_SAMPLES = 25000
VAL_SAMPLES = 2500

SEED = 42

MAX_SOURCE_LENGTH = 256
MAX_TARGET_LENGTH = 256


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("IndicTrans2 LoRA Fine-Tuning - FULL 25K")
print("=" * 70)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

    gpu_memory = (
        torch.cuda.get_device_properties(0).total_memory
        / (1024 ** 3)
    )

    print("GPU Memory:", round(gpu_memory, 2), "GB")

print("Training samples:", TRAIN_SAMPLES)
print("Validation samples:", VAL_SAMPLES)
print("Epochs: 2")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)

train_df = train_df.dropna(
    subset=["sanskrit", "english"]
).reset_index(drop=True)

val_df = val_df.dropna(
    subset=["sanskrit", "english"]
).reset_index(drop=True)

train_df = train_df.sample(
    n=min(TRAIN_SAMPLES, len(train_df)),
    random_state=SEED
).reset_index(drop=True)

val_df = val_df.sample(
    n=min(VAL_SAMPLES, len(val_df)),
    random_state=SEED
).reset_index(drop=True)

print("Training samples:", len(train_df))
print("Validation samples:", len(val_df))


# ============================================================
# IAST → DEVANAGARI
# ============================================================

print("\nConverting IAST -> Devanagari...")


def convert_to_devanagari(text):

    try:
        return sanscript.transliterate(
            str(text),
            sanscript.IAST,
            sanscript.DEVANAGARI
        )

    except Exception as e:

        print("Transliteration warning:", e)

        return str(text)


train_sanskrit = [
    convert_to_devanagari(text)
    for text in train_df["sanskrit"]
]

val_sanskrit = [
    convert_to_devanagari(text)
    for text in val_df["sanskrit"]
]


# ============================================================
# INDIC PROCESSOR
# ============================================================

print("\nInitializing IndicProcessor...")

processor = IndicProcessor(
    inference=True
)


# ============================================================
# CREATE DATASETS
# ============================================================

train_dataset = Dataset.from_dict(
    {
        "source": train_sanskrit,
        "target": train_df["english"].astype(str).tolist()
    }
)

val_dataset = Dataset.from_dict(
    {
        "source": val_sanskrit,
        "target": val_df["english"].astype(str).tolist()
    }
)


# ============================================================
# TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

print("Tokenizer loaded.")


# ============================================================
# MODEL
# ============================================================

print("\nLoading IndicTrans2 model...")

DTYPE = (
    torch.float16
    if torch.cuda.is_available()
    else torch.float32
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=DTYPE
)

print("Model loaded.")


# ============================================================
# TOKENIZATION
# ============================================================

def preprocess_function(examples):

    processed_sources = []

    for text in examples["source"]:

        processed = processor.preprocess_batch(
            [text],
            src_lang="san_Deva",
            tgt_lang="eng_Latn"
        )

        processed_sources.append(
            processed[0]
        )

    model_inputs = tokenizer(
        processed_sources,
        max_length=MAX_SOURCE_LENGTH,
        truncation=True
    )

    labels = tokenizer(
        text_target=examples["target"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


print("\nTokenizing training data...")

tokenized_train = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names,
    desc="Tokenizing train"
)

print("\nTokenizing validation data...")

tokenized_val = val_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=val_dataset.column_names,
    desc="Tokenizing validation"
)

print("Tokenization complete.")


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
# TRAINING ARGUMENTS
# ============================================================

training_args = Seq2SeqTrainingArguments(

    output_dir=OUTPUT_DIR,

    # ========================================================
    # ACTUAL TRAINING
    # ========================================================

    num_train_epochs=2,

    per_device_train_batch_size=1,

    per_device_eval_batch_size=1,

    gradient_accumulation_steps=4,

    learning_rate=2e-4,

    weight_decay=0.01,

    warmup_ratio=0.05,

    max_grad_norm=1.0,

    # ========================================================
    # GPU
    # ========================================================

    fp16=torch.cuda.is_available(),

    # ========================================================
    # EVALUATION
    # ========================================================
    # IMPORTANT:
    # Do NOT evaluate every 10 steps.
    # Validation takes several minutes.

    eval_strategy="epoch",

    # ========================================================
    # CHECKPOINT
    # ========================================================

    save_strategy="epoch",

    save_total_limit=2,

    load_best_model_at_end=True,

    metric_for_best_model="eval_loss",

    greater_is_better=False,

    # ========================================================
    # LOGGING
    # ========================================================

    logging_strategy="steps",

    logging_steps=100,

    # ========================================================
    # DATA LOADING
    # ========================================================

    dataloader_num_workers=0,

    # ========================================================
    # OTHER
    # ========================================================

    report_to="none",

    seed=SEED,

    remove_unused_columns=True
)


# ============================================================
# TRAINER
# ============================================================

trainer = Seq2SeqTrainer(

    model=model,

    args=training_args,

    train_dataset=tokenized_train,

    eval_dataset=tokenized_val,

    processing_class=tokenizer,

    data_collator=data_collator
)


# ============================================================
# START TRAINING
# ============================================================

print("\n")
print("=" * 70)
print("STARTING FULL 25K LoRA TRAINING")
print("=" * 70)

print("Training samples : 25,000")
print("Validation       : 2,500")
print("Epochs           : 2")
print("Batch size       : 1")
print("Gradient accum.  : 4")
print("Learning rate    : 2e-4")
print("LoRA rank        : 8")
print("=" * 70)

train_result = trainer.train()


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n")
print("=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

eval_results = trainer.evaluate()

print("\nValidation results:")

for key, value in eval_results.items():
    print(f"{key}: {value}")


# ============================================================
# SAVE FINAL ADAPTER
# ============================================================

print("\n")
print("=" * 70)
print("SAVING FINAL LoRA ADAPTER")
print("=" * 70)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

model.save_pretrained(
    OUTPUT_DIR
)

tokenizer.save_pretrained(
    OUTPUT_DIR
)

print("\nLoRA adapter saved to:")

print(OUTPUT_DIR)

print("\nTraining complete.")

print("=" * 70)