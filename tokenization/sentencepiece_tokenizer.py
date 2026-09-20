import pandas as pd
import sentencepiece as spm
import os

TRAIN_FILE = "dataset/final/train.csv"
OUTPUT_DIR = "dataset/processed/tokenizers"

os.makedirs(OUTPUT_DIR, exist_ok=True)

SANSKRIT_TEXT = os.path.join(OUTPUT_DIR, "sanskrit_corpus.txt")
ENGLISH_TEXT = os.path.join(OUTPUT_DIR, "english_corpus.txt")

SANSKRIT_MODEL = os.path.join(OUTPUT_DIR, "sanskrit")
ENGLISH_MODEL = os.path.join(OUTPUT_DIR, "english")

VOCAB_SIZE = 16000


def create_corpus():

    print("Loading training dataset...")

    df = pd.read_csv(TRAIN_FILE)

    print("Training sentences:", len(df))

    print("Creating Sanskrit corpus...")

    with open(SANSKRIT_TEXT, "w", encoding="utf-8") as f:
        for text in df["sanskrit"].dropna():
            text = str(text).strip()
            if text:
                f.write(text + "\n")

    print("Creating English corpus...")

    with open(ENGLISH_TEXT, "w", encoding="utf-8") as f:
        for text in df["english"].dropna():
            text = str(text).strip()
            if text:
                f.write(text + "\n")


def train_tokenizer(input_file, model_prefix):

    print(f"\nTraining tokenizer: {model_prefix}")

    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        character_coverage=1.0,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<bos>",
        eos_piece="<eos>",
        input_sentence_size=200000,
        shuffle_input_sentence=True
    )

    print("Tokenizer created.")


def test_tokenizer(model_file, sentence):

    tokenizer = spm.SentencePieceProcessor(
        model_file=model_file
    )

    pieces = tokenizer.encode(
        sentence,
        out_type=str
    )

    ids = tokenizer.encode(
        sentence,
        out_type=int
    )

    print("\nOriginal:")
    print(sentence)

    print("\nPieces:")
    print(pieces)

    print("\nIDs:")
    print(ids)


if __name__ == "__main__":

    print("=" * 70)
    print("SENTENCEPIECE TOKENIZER")
    print("=" * 70)

    create_corpus()

    train_tokenizer(
        SANSKRIT_TEXT,
        SANSKRIT_MODEL
    )

    train_tokenizer(
        ENGLISH_TEXT,
        ENGLISH_MODEL
    )

    print("\n" + "=" * 70)
    print("TESTING TOKENIZERS")
    print("=" * 70)

    test_tokenizer(
        SANSKRIT_MODEL + ".model",
        "rāmaḥ vanaṃ gacchati"
    )

    test_tokenizer(
        ENGLISH_MODEL + ".model",
        "Rama goes to the forest."
    )

    print("\n" + "=" * 70)
    print("TOKENIZER CREATION COMPLETE")
    print("=" * 70)