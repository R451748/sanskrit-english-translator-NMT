from datasets import load_dataset
import os

DATASET = "buddhist-nlp/mitrasamgraha-released-data-only"

OUTPUT_DIR = "dataset/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Downloading Mitrasamgraha...")

dataset = load_dataset(DATASET)

print("\nDataset downloaded successfully!")
print(dataset)

for split in dataset:
    output_file = os.path.join(OUTPUT_DIR, f"{split}.csv")
    dataset[split].to_csv(output_file, index=False)
    print(f"Saved: {output_file}")