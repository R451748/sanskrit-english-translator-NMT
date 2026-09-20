import pandas as pd
import os

FILES = {
    "train": "dataset/final/train.csv",
"validation": "dataset/final/validation.csv",
"test": "dataset/final/test.csv"
}

print("=" * 70)
print("SANSKRIT → ENGLISH DATASET QUALITY CHECK")
print("=" * 70)

datasets = {}

for name, path in FILES.items():

    print(f"\n{'=' * 70}")
    print(f"{name.upper()}")
    print(f"{'=' * 70}")

    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        continue

    df = pd.read_csv(path)

    datasets[name] = df

    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    # Missing values
    print("\nMissing values:")
    print(df.isnull().sum())

    # Empty strings
    sanskrit_empty = (
        df["sanskrit"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    english_empty = (
        df["english"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    print(f"\nEmpty Sanskrit: {sanskrit_empty:,}")
    print(f"Empty English: {english_empty:,}")

    # Exact duplicate pairs
    print(f"Exact duplicate rows: {df.duplicated().sum():,}")

    # Duplicate Sanskrit
    print(
        f"Duplicate Sanskrit sentences: "
        f"{df['sanskrit'].duplicated().sum():,}"
    )

    # Duplicate English
    print(
        f"Duplicate English sentences: "
        f"{df['english'].duplicated().sum():,}"
    )

    # Length statistics
    sanskrit_words = df["sanskrit"].fillna("").astype(str).str.split().str.len()
    english_words = df["english"].fillna("").astype(str).str.split().str.len()

    print("\nSanskrit word count:")
    print(f"  Average: {sanskrit_words.mean():.2f}")
    print(f"  Minimum: {sanskrit_words.min()}")
    print(f"  Maximum: {sanskrit_words.max()}")

    print("\nEnglish word count:")
    print(f"  Average: {english_words.mean():.2f}")
    print(f"  Minimum: {english_words.min()}")
    print(f"  Maximum: {english_words.max()}")

    # Samples
    print("\nSample records:")
    print(df[["sanskrit", "english"]].head(3).to_string(index=False))


# ============================================================
# DATASET LEAKAGE
# ============================================================

print(f"\n{'=' * 70}")
print("TRAIN / VALIDATION / TEST LEAKAGE CHECK")
print(f"{'=' * 70}")

if len(datasets) == 3:

    train = datasets["train"]
    validation = datasets["validation"]
    test = datasets["test"]

    train_sanskrit = set(train["sanskrit"].dropna().astype(str))
    val_sanskrit = set(validation["sanskrit"].dropna().astype(str))
    test_sanskrit = set(test["sanskrit"].dropna().astype(str))

    train_val = train_sanskrit & val_sanskrit
    train_test = train_sanskrit & test_sanskrit
    val_test = val_sanskrit & test_sanskrit

    print(f"Train ∩ Validation: {len(train_val):,}")
    print(f"Train ∩ Test:       {len(train_test):,}")
    print(f"Validation ∩ Test:  {len(val_test):,}")

    print("\nExpected:")
    print("All three values should ideally be 0.")

# ============================================================

print(f"\n{'=' * 70}")
print("QUALITY CHECK COMPLETE")
print(f"{'=' * 70}")