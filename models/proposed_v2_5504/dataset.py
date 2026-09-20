import json
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


# ============================================================
# PROJECT PATH
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

class MorphologyTranslationDataset(Dataset):

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

        # ----------------------------------------------------
        # Sanskrit source tokens
        # ----------------------------------------------------

        source = torch.tensor(
            item["source_tokens"],
            dtype=torch.long
        )

        # ----------------------------------------------------
        # English target tokens
        # ----------------------------------------------------

        target = torch.tensor(
            item["target_tokens"],
            dtype=torch.long
        )

        # ----------------------------------------------------
        # Morphology tag IDs
        # ----------------------------------------------------

        morphology = torch.tensor(
            item["morphology_tag_ids"],
            dtype=torch.long
        )

        return {
            "source": source,
            "target": target,
            "morphology": morphology,
            "index": item["index"]
        }


# ============================================================
# COLLATE FUNCTION
# ============================================================

def morphology_collate_fn(batch):

    # --------------------------------------------------------
    # Source
    # --------------------------------------------------------

    sources = torch.stack(
        [
            item["source"]
            for item in batch
        ]
    )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    targets = torch.stack(
        [
            item["target"]
            for item in batch
        ]
    )

    # --------------------------------------------------------
    # Morphology
    #
    # Morphology sequences have different lengths.
    # PAD ID = 0
    # --------------------------------------------------------

    max_morph_length = max(
        item["morphology"].size(0)
        for item in batch
    )

    morphology = torch.zeros(
        len(batch),
        max_morph_length,
        dtype=torch.long
    )

    for i, item in enumerate(batch):

        length = item["morphology"].size(0)

        morphology[
            i,
            :length
        ] = item["morphology"]

    # --------------------------------------------------------
    # Original dataset indices
    # --------------------------------------------------------

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
        "morphology": morphology,
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
        collate_fn=morphology_collate_fn,
        pin_memory=torch.cuda.is_available()
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("TESTING V2-5504 DATASET")
    print("=" * 70)

    print("\nProject root:")
    print(PROJECT_ROOT)

    print("\nDataset file:")
    print(DATA_FILE)

    print("\nDataset file exists:")
    print(DATA_FILE.exists())

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    dataset = MorphologyTranslationDataset()

    print(
        f"\nDataset size: {len(dataset)}"
    )

    # --------------------------------------------------------
    # Create dataloader
    # --------------------------------------------------------

    loader = create_dataloader(
        dataset,
        batch_size=8,
        shuffle=False
    )

    # --------------------------------------------------------
    # Get first batch
    # --------------------------------------------------------

    batch = next(iter(loader))

    # --------------------------------------------------------
    # Shapes
    # --------------------------------------------------------

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
        "Morphology:",
        batch["morphology"].shape
    )

    print(
        "Indices:",
        batch["indices"].tolist()
    )

    # --------------------------------------------------------
    # First example
    # --------------------------------------------------------

    print("\nFirst morphology IDs:")

    print(
        batch["morphology"][0].tolist()
    )

    print("\nFirst source tokens:")

    print(
        batch["source"][0].tolist()
    )

    print("\nFirst target tokens:")

    print(
        batch["target"][0].tolist()
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATASET TEST COMPLETE")
    print("=" * 70)