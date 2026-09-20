import pandas as pd
import os
import re
import unicodedata

INPUT_DIR = "dataset/raw"
OUTPUT_DIR = "dataset/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_text(text):
    """Normalize and clean Sanskrit/English text."""

    if pd.isna(text):
        return ""

    text = str(text)

    # Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)

    # Replace newlines/tabs with spaces
    text = re.sub(r"[\r\n\t]+", " ", text)

    # Normalize multiple spaces
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing spaces
    text = text.strip()

    return text


def process_file(filename):

    input_path = os.path.join(INPUT_DIR, filename)
    output_path = os.path.join(OUTPUT_DIR, filename)

    print(f"\nProcessing: {input_path}")

    df = pd.read_csv(input_path)

    original_rows = len(df)

    # Clean both columns
    df["sanskrit"] = df["sanskrit"].apply(clean_text)
    df["english"] = df["english"].apply(clean_text)

    # Remove empty rows
    df = df[
        (df["sanskrit"] != "") &
        (df["english"] != "")
    ]

    after_empty = len(df)

    # Remove exact duplicate translation pairs
    df = df.drop_duplicates(
        subset=["sanskrit", "english"]
    )

    after_duplicates = len(df)

    # Reset index
    df = df.reset_index(drop=True)

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8"
    )

    print("Original rows:", original_rows)
    print("After empty removal:", after_empty)
    print("After duplicate removal:", after_duplicates)
    print("Removed:", original_rows - after_duplicates)
    print("Saved:", output_path)


if __name__ == "__main__":

    print("=" * 70)
    print("DATASET CLEANING")
    print("=" * 70)

    process_file("train.csv")
    process_file("validation.csv")
    process_file("test.csv")

    print("\n" + "=" * 70)
    print("CLEANING COMPLETE")
    print("=" * 70)