import os
import json
import time
import requests
import pandas as pd
from tqdm import tqdm

from indic_transliteration import sanscript


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "dataset/final/train.csv"
OUTPUT_DIR = "morphology/batches"

BATCH_SIZE = 100
NUM_TEST_SENTENCES = 10

API_URL = "https://sanskrit.inria.fr/cgi-bin/SKT/sktgraph.cgi"


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# FIX MOJIBAKE
# ============================================================

def fix_mojibake(text):

    if not isinstance(text, str):
        return text

    try:

        # Detect common UTF-8 -> Latin-1 mojibake
        if any(
            x in text
            for x in [
                "Ã",
                "Å",
                "á",
                "Ä",
                "â"
            ]
        ):

            fixed = text.encode(
                "latin1"
            ).decode(
                "utf-8"
            )

            return fixed

    except (
        UnicodeEncodeError,
        UnicodeDecodeError
    ):
        pass

    return text


# ============================================================
# CONVERT IAST -> VELTHUIS
# ============================================================

def iast_to_velthuis(text):

    return sanscript.transliterate(
        text,
        sanscript.IAST,
        sanscript.VELTHUIS
    )


# ============================================================
# MORPHOLOGY ANALYSIS
# ============================================================

def analyze_sentence(sentence):

    try:

        # Fix encoding
        unicode_text = fix_mojibake(
            sentence
        )

        # Convert IAST -> Velthuis
        velthuis_text = iast_to_velthuis(
            unicode_text
        )

        # INRIA Sanskrit graph/parser
        params = {
            "t": "VH",
            "lex": "SH",
            "font": "roma",
            "cache": "f",
            "st": "t",
            "us": "f",
            "text": velthuis_text,
            "topic": "",
            "abs": "f",
            "corpmode": "",
            "corpdir": "",
            "sentno": "",
            "mode": "o"
        }

        response = requests.get(
            API_URL,
            params=params,
            timeout=30
        )

        response.raise_for_status()

        html = response.text

        # Check whether the server rejected input
        if (
            "Wrong input" in html
            or "Undefined token" in html
            or "Unknown transliteration" in html
        ):

            return {
                "original_sentence": sentence,
                "unicode_sentence": unicode_text,
                "velthuis_sentence": velthuis_text,
                "status": "analyzer_error",
                "response": html[:2000]
            }

        return {
            "original_sentence": sentence,
            "unicode_sentence": unicode_text,
            "velthuis_sentence": velthuis_text,
            "status": "success",
            "response": html[:10000]
        }

    except Exception as e:

        return {
            "original_sentence": sentence,
            "status": "request_error",
            "response": "",
            "error": str(e)
        }


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("SANSKRIT MORPHOLOGY TEST")
print("=" * 70)

df = pd.read_csv(
    INPUT_FILE
)

print(
    f"Dataset size: {len(df)}"
)

print(
    f"Testing: {NUM_TEST_SENTENCES} sentences"
)

print()


# ============================================================
# TEST FIRST 10 SENTENCES
# ============================================================

test_df = df.head(
    NUM_TEST_SENTENCES
)

results = []


for index, row in tqdm(
    test_df.iterrows(),
    total=len(test_df)
):

    result = analyze_sentence(
        row["sanskrit"]
    )

    result["english"] = row[
        "english"
    ]

    results.append(
        result
    )

    time.sleep(0.5)


# ============================================================
# SAVE
# ============================================================

output_file = os.path.join(
    OUTPUT_DIR,
    "morphology_test.json"
)

with open(
    output_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# SUMMARY
# ============================================================

success = sum(
    r["status"] == "success"
    for r in results
)

errors = len(results) - success

print()
print("=" * 70)
print("TEST COMPLETED")
print("=" * 70)

print(
    f"Successful analyses: {success}/{len(results)}"
)

print(
    f"Errors: {errors}/{len(results)}"
)

print(
    f"Saved: {output_file}"
)

print()


# ============================================================
# DISPLAY CONVERSIONS
# ============================================================

for i, result in enumerate(
    results[:5],
    start=1
):

    print(
        f"\nExample {i}"
    )

    print(
        "Original:"
    )
    print(
        result.get(
            "original_sentence",
            ""
        )
    )

    print(
        "Unicode:"
    )
    print(
        result.get(
            "unicode_sentence",
            ""
        )
    )

    print(
        "Velthuis:"
    )
    print(
        result.get(
            "velthuis_sentence",
            ""
        )
    )

    print(
        "Status:"
    )
    print(
        result["status"]
    )