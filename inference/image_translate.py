from pathlib import Path
import sys
import json
import subprocess

import torch
import sentencepiece as spm

from indic_transliteration import sanscript

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.proposed_v2_5504.model import (
    MorphologyConditionedTransformer
)


# ============================================================
# CONFIGURATION
# ============================================================

TESSERACT_PATH = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

TESSDATA_PATH = (
    r"C:\Program Files\Tesseract-OCR\tessdata"
)

MORPH_MODEL_NAME = "chronbmm/sanskrit5-multitask"

CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "proposed_v2_5504"
    / "checkpoints"
    / "best_model.pt"
)

SANSKRIT_TOKENIZER = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "tokenizers"
    / "sanskrit.model"
)

ENGLISH_TOKENIZER = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "tokenizers"
    / "english.model"
)

MORPH_VOCAB = (
    PROJECT_ROOT
    / "morphology"
    / "morphology_tag_vocab.json"
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# IMPORT MORPHOLOGY MODEL
# ============================================================

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)


# ============================================================
# LOAD MORPHOLOGY MODEL
# ============================================================

print("Loading local morphology model...")

morph_tokenizer = AutoTokenizer.from_pretrained(
    MORPH_MODEL_NAME
)

morph_model = AutoModelForSeq2SeqLM.from_pretrained(
    MORPH_MODEL_NAME
)

morph_model = morph_model.to(DEVICE)
morph_model.eval()


# ============================================================
# LOAD MORPHOLOGY VOCABULARY
# ============================================================

with open(
    MORPH_VOCAB,
    "r",
    encoding="utf-8"
) as f:

    morphology_vocab = json.load(f)


print(
    "Morphology vocabulary:",
    len(morphology_vocab)
)


# ============================================================
# LOAD SENTENCEPIECE TOKENIZERS
# ============================================================

src_sp = spm.SentencePieceProcessor()
src_sp.load(
    str(SANSKRIT_TOKENIZER)
)

tgt_sp = spm.SentencePieceProcessor()
tgt_sp.load(
    str(ENGLISH_TOKENIZER)
)


# ============================================================
# LOAD V2 MODEL
# ============================================================

print("Loading V2-5504 Transformer...")

model = MorphologyConditionedTransformer(
    src_vocab_size=16000,
    tgt_vocab_size=16000,
    morph_vocab_size=392,
    d_model=256,
    nhead=4,
    num_encoder_layers=3,
    num_decoder_layers=3,
    dim_feedforward=1024,
    dropout=0.1
).to(DEVICE)


checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    "Checkpoint epoch:",
    checkpoint.get("epoch")
)

print(
    "Validation loss:",
    checkpoint.get("val_loss")
)


# ============================================================
# OCR
# ============================================================

def extract_sanskrit_text(image_path):

    command = [
        TESSERACT_PATH,
        "--tessdata-dir",
        TESSDATA_PATH,
        str(image_path),
        "stdout",
        "-l",
        "san",
        "--psm",
        "6"
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8"
    )

    if result.returncode != 0:

        raise RuntimeError(
            result.stderr
        )

    return result.stdout.strip()


# ============================================================
# DEVANAGARI → IAST
# ============================================================

def devanagari_to_iast(text):

    return sanscript.transliterate(
        text,
        sanscript.DEVANAGARI,
        sanscript.IAST
    )


# ============================================================
# MORPHOLOGY
# ============================================================

@torch.no_grad()
def analyze_morphology(text):

    prompt = "SLM " + text

    inputs = morph_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    outputs = morph_model.generate(
        **inputs,
        max_new_tokens=128,
        num_beams=2
    )

    result = morph_tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return result.strip()


# ============================================================
# CONVERT MORPHOLOGY OUTPUT → TAG IDS
# ============================================================

# def morphology_to_ids(morphology_text):

#     tag_ids = []

#     tokens = morphology_text.split()

#     for token in tokens:

#         # Expected:
#         # surface_lemma_TAG

#         parts = token.rsplit(
#             "_",
#             1
#         )

#         if len(parts) != 2:
#             continue

#         tag = parts[1].strip()

#         if not tag:
#             continue

#         tag_id = morphology_vocab.get(
#             tag,
#             morphology_vocab["<UNK>"]
#         )

#         tag_ids.append(tag_id)

#     # Never give the model an empty morphology sequence
#     if not tag_ids:

#         tag_ids = [
#             morphology_vocab["<UNK>"]
#         ]

#     # Maximum morphology length
#     tag_ids = tag_ids[:128]

#     return tag_ids

def morphology_to_ids(morphology_text):

    tag_ids = []

    tokens = morphology_text.split()

    for token in tokens:

        # Expected:
        # surface_lemma_TAG

        parts = token.rsplit("_", 1)

        if len(parts) != 2:
            tag_ids.append(
                morphology_vocab["<UNK>"]
            )
            continue

        tag = parts[1].strip()

        # Empty or missing tag → <UNK>
        if not tag:
            tag_ids.append(
                morphology_vocab["<UNK>"]
            )
            continue

        tag_id = morphology_vocab.get(
            tag,
            morphology_vocab["<UNK>"]
        )

        tag_ids.append(tag_id)

    # Never give the model an empty morphology sequence
    if not tag_ids:
        tag_ids = [
            morphology_vocab["<UNK>"]
        ]

    # Maximum morphology length
    tag_ids = tag_ids[:128]

    return tag_ids
# ============================================================
# TOKENIZE SANSKRIT
# ============================================================

def tokenize_sanskrit(text):

    token_ids = src_sp.encode(
        text,
        out_type=int
    )

    # Reserve space for EOS
    token_ids = token_ids[:127]

    token_ids.append(3)

    return token_ids


# ============================================================
# TRANSLATION
# ============================================================

@torch.no_grad()
def translate(
    source_ids,
    morphology_ids,
    max_length=100
):

    source = torch.tensor(
        [source_ids],
        dtype=torch.long,
        device=DEVICE
    )

    morphology = torch.tensor(
        [morphology_ids],
        dtype=torch.long,
        device=DEVICE
    )

    # BOS = 2
    generated = torch.tensor(
        [[2]],
        dtype=torch.long,
        device=DEVICE
    )

    for _ in range(max_length):

        # Model expects tgt with at least 2 positions
        # because forward() uses tgt[:, :-1].
        temporary_target = torch.cat(
            [
                generated,
                torch.tensor(
                    [[0]],
                    dtype=torch.long,
                    device=DEVICE
                )
            ],
            dim=1
        )

        output = model(
            source,
            morphology,
            temporary_target
        )

        next_token = output[
            :,
            -1,
            :
        ].argmax(
            dim=-1
        )

        token_id = next_token.item()

        generated = torch.cat(
            [
                generated,
                next_token.unsqueeze(1)
            ],
            dim=1
        )

        # EOS
        if token_id == 3:
            break

    return generated[0].tolist()


# ============================================================
# DECODE ENGLISH
# ============================================================

def decode_english(token_ids):

    # Remove BOS
    if (
        token_ids
        and token_ids[0] == 2
    ):
        token_ids = token_ids[1:]

    # Stop at EOS
    if 3 in token_ids:

        token_ids = token_ids[
            :token_ids.index(3)
        ]

    # Remove PAD
    token_ids = [
        token
        for token in token_ids
        if token != 0
    ]

    if not token_ids:
        return ""

    return tgt_sp.decode(
        token_ids
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    if len(sys.argv) < 2:

        print(
            r"Usage: python inference\image_translate.py ocr\images\img1.png"
        )

        return

    image_path = Path(
        sys.argv[1]
    )

    if not image_path.exists():

        print(
            "Image not found:",
            image_path
        )

        return

    print("\n" + "=" * 70)
    print("SANSKRIT IMAGE → ENGLISH TRANSLATION")
    print("=" * 70)

    # --------------------------------------------------------
    # OCR
    # --------------------------------------------------------

    print("\n[1/5] OCR...")

    sanskrit = extract_sanskrit_text(
        image_path
    )

    print(
        "Sanskrit:",
        sanskrit
    )

    # --------------------------------------------------------
    # IAST
    # --------------------------------------------------------

    print("\n[2/5] Devanagari → IAST...")

    iast = devanagari_to_iast(
        sanskrit
    )

    print(
        "IAST:",
        iast
    )

    # --------------------------------------------------------
    # Morphology
    # --------------------------------------------------------

    print("\n[3/5] Morphology...")

    morphology = analyze_morphology(
        iast
    )

    print(
        "Morphology:",
        morphology
    )

    # --------------------------------------------------------
    # Morphology IDs
    # --------------------------------------------------------

    morphology_ids = morphology_to_ids(
        morphology
    )

    print(
        "Morphology IDs:",
        morphology_ids
    )

    # --------------------------------------------------------
    # Sanskrit tokens
    # --------------------------------------------------------

    print("\n[4/5] Sanskrit tokenization...")

    source_ids = tokenize_sanskrit(
        iast
    )

    print(
        "Source tokens:",
        len(source_ids)
    )

    # --------------------------------------------------------
    # Translation
    # --------------------------------------------------------

    print("\n[5/5] V2 translation...")

    english_ids = translate(
        source_ids,
        morphology_ids
    )

    english = decode_english(
        english_ids
    )

    print("\n" + "=" * 70)
    print("FINAL ENGLISH TRANSLATION")
    print("=" * 70)

    print("\n" + english)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()