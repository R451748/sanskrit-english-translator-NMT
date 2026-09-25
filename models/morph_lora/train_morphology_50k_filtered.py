import os
import json
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_MODEL = "krpraveen/indictrans2-sanskrit-en-finetuned"

TRAIN_FILE = (
    "models/morph_lora/morphology_50k_filtered_split/train.json"
)

VAL_FILE = (
    "models/morph_lora/morphology_50k_filtered_split/validation.json"
)

OUTPUT_DIR = (
    "models/morph_lora/morphology_50k_filtered"
)

MAX_SOURCE_LENGTH = 256
MAX_TARGET_LENGTH = 128


# ============================================================
# GPU CHECK
# ============================================================

print("=" * 70)
print("MORPHOLOGY 50K FILTERED — FULL LoRA TRAINING")
print("=" * 70)

print()

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. Stop training and check the GPU environment."
    )

print("GPU:", torch.cuda.get_device_name(0))

gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

print(f"GPU memory: {gpu_memory:.2f} GB")


# ============================================================
# LOAD DATA
# ============================================================

print()
print("Loading datasets...")

with open(TRAIN_FILE, "r", encoding="utf-8") as f:
    train_data = json.load(f)

with open(VAL_FILE, "r", encoding="utf-8") as f:
    val_data = json.load(f)

print(f"Training records:   {len(train_data):,}")
print(f"Validation records: {len(val_data):,}")


# ============================================================
# BASIC DATA CHECK
# ============================================================

if len(train_data) != 35506:
    print(
        f"WARNING: expected approximately 35,506 training records, "
        f"found {len(train_data):,}"
    )

if len(val_data) != 3946:
    print(
        f"WARNING: expected approximately 3,946 validation records, "
        f"found {len(val_data):,}"
    )


# ============================================================
# CHECK MORPHOLOGY
# ============================================================

empty_train = sum(
    not x.get("morphology", "").strip()
    for x in train_data
)

empty_val = sum(
    not x.get("morphology", "").strip()
    for x in val_data
)

print()
print("Empty morphology records:")
print("Train:", empty_train)
print("Validation:", empty_val)

if empty_train > 0 or empty_val > 0:
    raise RuntimeError(
        "Empty morphology records detected. Stop training."
    )


# ============================================================
# HUGGING FACE DATASETS
# ============================================================

train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)


# ============================================================
# TOKENIZER
# ============================================================

print()
print("Loading IndicTrans2 tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
)


# ============================================================
# SOURCE CONSTRUCTION
# ============================================================

def build_source(example):

    sanskrit = example["sanskrit"].strip()
    morphology = example["morphology"].strip()

    source = (
        "san_Deva eng_Latn "
        + sanskrit
        + " MORPHOLOGY "
        + morphology
    )

    return source


# ============================================================
# TOKENIZATION
# ============================================================

def preprocess(example):

    source = build_source(example)

    target = example["english"].strip()

    model_inputs = tokenizer(
        source,
        max_length=MAX_SOURCE_LENGTH,
        truncation=True,
    )

    labels = tokenizer(
        text_target=target,
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


print()
print("Tokenizing training dataset...")

train_tokenized = train_dataset.map(
    preprocess,
    remove_columns=train_dataset.column_names,
    desc="Train tokenization",
)

print()
print("Tokenizing validation dataset...")

val_tokenized = val_dataset.map(
    preprocess,
    remove_columns=val_dataset.column_names,
    desc="Validation tokenization",
)


# ============================================================
# MODEL
# ============================================================

print()
print("Loading IndicTrans2 model...")

model = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    torch_dtype=torch.float16,
)


# ============================================================
# LoRA CONFIGURATION
# ============================================================

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
    ],

    bias="none",

    task_type=TaskType.SEQ_2_SEQ_LM,
)


# ============================================================
# APPLY LoRA
# ============================================================

model = get_peft_model(
    model,
    lora_config,
)

print()
print("LoRA parameters:")

model.print_trainable_parameters()


# ============================================================
# DATA COLLATOR
# ============================================================

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
)


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

training_args = Seq2SeqTrainingArguments(

    output_dir=OUTPUT_DIR,

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    num_train_epochs=2,

    per_device_train_batch_size=1,

    per_device_eval_batch_size=1,

    gradient_accumulation_steps=4,

    # --------------------------------------------------------
    # Optimization
    # --------------------------------------------------------

    learning_rate=2e-4,

    weight_decay=0.01,

    warmup_ratio=0.05,

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    fp16=True,

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    logging_steps=100,

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    eval_strategy="epoch",

    # --------------------------------------------------------
    # Checkpoints
    # --------------------------------------------------------

    save_strategy="epoch",

    save_total_limit=2,

    # --------------------------------------------------------
    # Best model
    # --------------------------------------------------------

    load_best_model_at_end=True,

    metric_for_best_model="eval_loss",

    greater_is_better=False,

    # --------------------------------------------------------
    # Data loading
    # --------------------------------------------------------

    dataloader_num_workers=0,

    # --------------------------------------------------------
    # Reporting
    # --------------------------------------------------------

    report_to="none",

    # --------------------------------------------------------
    # PEFT compatibility
    # --------------------------------------------------------

    remove_unused_columns=False,
)


# ============================================================
# TRAINER
# ============================================================

trainer = Seq2SeqTrainer(

    model=model,

    args=training_args,

    train_dataset=train_tokenized,

    eval_dataset=val_tokenized,

    processing_class=tokenizer,

    data_collator=data_collator,
)


# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 70)
print("STARTING FULL TRAINING")
print("=" * 70)

print()
print("Train records:       ", f"{len(train_data):,}")
print("Validation records:  ", f"{len(val_data):,}")
print("Epochs:               2")
print("Batch size:           1")
print("Gradient accumulation: 4")
print("Learning rate:        0.0002")
print("LoRA rank:            8")
print("LoRA alpha:           16")
print("FP16:                 True")

print()
print("Output directory:")
print(OUTPUT_DIR)

print()
print("=" * 70)


train_result = trainer.train()


# ============================================================
# FINAL EVALUATION
# ============================================================

print()
print("=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

eval_results = trainer.evaluate()

for key, value in eval_results.items():

    if isinstance(value, float):
        print(f"{key}: {value:.6f}")
    else:
        print(f"{key}: {value}")


# ============================================================
# SAVE MODEL
# ============================================================

print()
print("=" * 70)
print("SAVING MODEL")
print("=" * 70)

trainer.save_model(OUTPUT_DIR)

tokenizer.save_pretrained(OUTPUT_DIR)


# ============================================================
# SAVE TRAINING RESULTS
# ============================================================

results_file = os.path.join(
    OUTPUT_DIR,
    "training_results.json"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

results = {
    "base_model": BASE_MODEL,

    "train_records": len(train_data),

    "validation_records": len(val_data),

    "epochs": 2,

    "batch_size": 1,

    "gradient_accumulation_steps": 4,

    "learning_rate": 2e-4,

    "weight_decay": 0.01,

    "warmup_ratio": 0.05,

    "fp16": True,

    "max_source_length": MAX_SOURCE_LENGTH,

    "max_target_length": MAX_TARGET_LENGTH,

    "lora_r": 8,

    "lora_alpha": 16,

    "lora_dropout": 0.05,

    "train_result": {
        k: (
            float(v)
            if isinstance(v, (int, float))
            else str(v)
        )
        for k, v in train_result.metrics.items()
    },

    "eval_result": {
        k: (
            float(v)
            if isinstance(v, (int, float))
            else str(v)
        )
        for k, v in eval_results.items()
    },
}


with open(
    results_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=2
    )


# ============================================================
# DONE
# ============================================================

print()
print("=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print()
print("Model:")
print(OUTPUT_DIR)

print()
print("Results:")
print(results_file)

print()
print("Do NOT evaluate against the test set yet.")
print("Next step will be the controlled 5,056-sentence evaluation.")

print("=" * 70)