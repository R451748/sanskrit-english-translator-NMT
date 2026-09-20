import json
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


# ============================================================
# PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "morphology_train_5504.json"
)


# ============================================================
# DATASET
# ============================================================

class TranslationDataset(Dataset):

    def __init__(self, data_file=DATA_FILE):

        data_file = Path(data_file)

        if not data_file.exists():
            raise FileNotFoundError(
                f"Dataset file not found:\n{data_file}"
            )

        with open(
            data_file,
            "r",
            encoding="utf-8"
        ) as f:

            self.data = json.load(f)

        print(
            f"Loaded dataset: {len(self.data)} records"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        item = self.data[idx]

        source = torch.tensor(
            item["source_tokens"],
            dtype=torch.long
        )

        target = torch.tensor(
            item["target_tokens"],
            dtype=torch.long
        )

        return {
            "source": source,
            "target": target,
            "index": item["index"]
        }


# ============================================================
# COLLATE
# ============================================================

def collate_fn(batch):

    sources = torch.stack(
        [
            item["source"]
            for item in batch
        ]
    )

    targets = torch.stack(
        [
            item["target"]
            for item in batch
        ]
    )

    indices = torch.tensor(
        [
            item["index"]
            for item in batch
        ],
        dtype=torch.long
    )

    return {
        "source": sources,
        "target": targets,
        "indices": indices
    }


# ============================================================
# DATALOADER
# ============================================================

def create_dataloader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0
):

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("TESTING BASELINE-5504 DATASET")
    print("=" * 70)

    print("\nProject root:")
    print(PROJECT_ROOT)

    print("\nDataset:")
    print(DATA_FILE)

    print("\nExists:")
    print(DATA_FILE.exists())

    dataset = TranslationDataset()

    print(
        "\nDataset size:",
        len(dataset)
    )

    loader = create_dataloader(
        dataset,
        batch_size=8,
        shuffle=False
    )

    batch = next(iter(loader))

    print("\nBatch shapes:")

    print(
        "Source:",
        batch["source"].shape
    )

    print(
        "Target:",
        batch["target"].shape
    )

    print(
        "Indices:",
        batch["indices"].tolist()
    )

    print("\n" + "=" * 70)
    print("BASELINE DATASET TEST COMPLETE")
    print("=" * 70)