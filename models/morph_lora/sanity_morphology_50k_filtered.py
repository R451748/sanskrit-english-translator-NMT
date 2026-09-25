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

# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_MODEL = "krpraveen/indictrans2-sanskrit-en-finetuned"

TRAIN_FILE = (
    "models/morph_lora/morphology_50k_filtered_split/train.json"
)

OUTPUT_DIR = (
    "models/morph_lora/sanity_50k_filtered"
)

MAX_SOURCE_LENGTH = 256
MAX_TARGET_LENGTH = 128

SANITY_SAMPLES = 20


# --------------------------------------------------
# Device
# --------------------------------------------------

print("=" * 70)
print("MORPHOLOGY 50K FILTERED — SANITY TEST")
print("=" * 70)

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# --------------------------------------------------
# Load data
# --------------------------------------------------

with open(TRAIN_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

data = data[:SANITY_SAMPLES]

print()
print("Sanity samples:", len(data))


# --------------------------------------------------
# Show first example
# --------------------------------------------------

print()
print("Example Sanskrit:")
print(data[0]["sanskrit"])

print()
print("Example morphology:")
print(data[0]["morphology"][:500])

print()
print("Example English:")
print(data[0]["english"])


# --------------------------------------------------
# Dataset
# --------------------------------------------------

dataset = Dataset.from_list(data)


# --------------------------------------------------
# Tokenizer
# --------------------------------------------------

print()
print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
)


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

def preprocess(example):

    source = (
        "san_Deva eng_Latn "
        + example["sanskrit"].strip()
        + " MORPHOLOGY "
        + example["morphology"].strip()
    )

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
print("Tokenizing...")

tokenized = dataset.map(
    preprocess,
    remove_columns=dataset.column_names,
)


# --------------------------------------------------
# Model
# --------------------------------------------------

print()
print("Loading IndicTrans2 model...")

model = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    torch_dtype=torch.float16,
)


# --------------------------------------------------
# LoRA
# --------------------------------------------------

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

model = get_peft_model(
    model,
    lora_config,
)

print()
model.print_trainable_parameters()


# --------------------------------------------------
# Data collator
# --------------------------------------------------

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
)


# --------------------------------------------------
# Sanity training
# --------------------------------------------------

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,

    max_steps=20,

    per_device_train_batch_size=1,

    gradient_accumulation_steps=1,

    learning_rate=2e-4,

    fp16=True,

    logging_steps=1,

    save_strategy="no",

    eval_strategy="no",

    report_to="none",

    dataloader_num_workers=0,

    remove_unused_columns=False,
)


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,

    train_dataset=tokenized,

    processing_class=tokenizer,

    data_collator=data_collator,
)


# --------------------------------------------------
# Train
# --------------------------------------------------

print()
print("=" * 70)
print("STARTING 20-STEP SANITY TRAINING")
print("=" * 70)

result = trainer.train()


# --------------------------------------------------
# Result
# --------------------------------------------------

print()
print("=" * 70)
print("SANITY TEST COMPLETED")
print("=" * 70)

print("Training loss:", result.training_loss)

print()
print("If training completed without CUDA/tokenizer/source-format errors,")
print("the filtered morphology dataset is ready for the full experiment.")

print("=" * 70)