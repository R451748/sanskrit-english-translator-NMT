from pathlib import Path
import sys

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from indic_transliteration import sanscript


# ============================================================
# LOCAL MORPHOLOGY MODEL
# ============================================================

MODEL_NAME = "chronbmm/sanskrit5-multitask"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading local Sanskrit morphology model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME
)

model = model.to(DEVICE)
model.eval()

print("Device:", DEVICE)


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

def analyze_morphology(sanskrit_iast):

    prompt = "SLM " + sanskrit_iast

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            num_beams=2
        )

    result = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return result.strip()


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) < 2:

        print(
            r'Usage: python inference\image_morphology.py "Sanskrit text"'
        )

        return

    devanagari_text = " ".join(
        sys.argv[1:]
    )

    print("=" * 70)
    print("SANSKRIT MORPHOLOGY")
    print("=" * 70)

    print("\nOCR Sanskrit:")
    print(devanagari_text)

    # --------------------------------------------------------
    # Transliteration
    # --------------------------------------------------------

    iast_text = devanagari_to_iast(
        devanagari_text
    )

    print("\nIAST:")
    print(iast_text)

    # --------------------------------------------------------
    # Morphology
    # --------------------------------------------------------

    print("\nAnalyzing morphology...")

    try:

        morphology = analyze_morphology(
            iast_text
        )

        print("\nMorphology:")
        print(morphology)

    except Exception as e:

        print("\nMORPHOLOGY ERROR:")
        print(e)


if __name__ == "__main__":
    main()