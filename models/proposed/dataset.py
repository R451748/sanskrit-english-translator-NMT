import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIGURATION
# ============================================================

MORPHOLOGY_FILE = (
    "dataset/processed/morphology_train_2720.json"
)

SOURCE_FILE = (
    "dataset/processed/sequences/train_source.npy"
)

TARGET_FILE = (
    "dataset/processed/sequences/train_target.npy"
)

PAD_ID = 0


# ============================================================
# DATASET
# ============================================================

class MorphologyTranslationDataset(Dataset):

    def __init__(
        self,
        source_sequences,
        target_sequences,
        morphology_sequences
    ):

        assert len(source_sequences) == len(
            target_sequences
        )

        assert len(source_sequences) == len(
            morphology_sequences
        )

        self.source = source_sequences
        self.target = target_sequences
        self.morphology = morphology_sequences

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):

        return {
            "source": torch.tensor(
                self.source[index],
                dtype=torch.long
            ),

            "target": torch.tensor(
                self.target[index],
                dtype=torch.long
            ),

            "morphology": torch.tensor(
                self.morphology[index],
                dtype=torch.long
            )
        }


# ============================================================
# DYNAMIC MORPHOLOGY PADDING
# ============================================================

def morphology_collate(batch):

    sources = [
        item["source"]
        for item in batch
    ]

    targets = [
        item["target"]
        for item in batch
    ]

    morphology = [
        item["morphology"]
        for item in batch
    ]

    # --------------------------------------------------------
    # Source
    # --------------------------------------------------------

    max_source_len = max(
        x.size(0)
        for x in sources
    )

    padded_sources = torch.full(
        (
            len(sources),
            max_source_len
        ),
        PAD_ID,
        dtype=torch.long
    )

    for i, source in enumerate(sources):

        length = source.size(0)

        padded_sources[
            i,
            :length
        ] = source

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    max_target_len = max(
        x.size(0)
        for x in targets
    )

    padded_targets = torch.full(
        (
            len(targets),
            max_target_len
        ),
        PAD_ID,
        dtype=torch.long
    )

    for i, target in enumerate(targets):

        length = target.size(0)

        padded_targets[
            i,
            :length
        ] = target

    # --------------------------------------------------------
    # Morphology
    # --------------------------------------------------------

    max_morphology_len = max(
        x.size(0)
        for x in morphology
    )

    padded_morphology = torch.full(
        (
            len(morphology),
            max_morphology_len
        ),
        PAD_ID,
        dtype=torch.long
    )

    for i, morph in enumerate(morphology):

        length = morph.size(0)

        padded_morphology[
            i,
            :length
        ] = morph

    return {
        "source": padded_sources,
        "target": padded_targets,
        "morphology": padded_morphology
    }


# ============================================================
# LOAD MORPHOLOGY DATA
# ============================================================

def load_morphology_sequences():

    print("Loading morphology data...")

    with open(
        MORPHOLOGY_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    records = data["records"]

    morphology_sequences = []

    for record in records:

        tags = record.get(
            "morphology_tag_ids",
            []
        )

        morphology_sequences.append(
            tags
        )

    print(
        f"Morphology records: "
        f"{len(morphology_sequences):,}"
    )

    return morphology_sequences


# ============================================================
# CREATE DATASET
# ============================================================

def create_dataset():

    print("=" * 70)
    print("MORPHOLOGY-AWARE DATASET")
    print("=" * 70)

    # --------------------------------------------------------
    # Load existing tokenized Sanskrit
    # --------------------------------------------------------

    source = np.load(
        SOURCE_FILE
    )

    # --------------------------------------------------------
    # Load existing tokenized English
    # --------------------------------------------------------

    target = np.load(
        TARGET_FILE
    )

    # --------------------------------------------------------
    # Only use first 2720 rows
    # --------------------------------------------------------

    source = source[:2720]

    target = target[:2720]

    # --------------------------------------------------------
    # Morphology
    # --------------------------------------------------------

    morphology = load_morphology_sequences()

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    assert len(source) == len(
        morphology
    ), (
        f"Source count {len(source)} "
        f"!= morphology count "
        f"{len(morphology)}"
    )

    assert len(target) == len(
        morphology
    ), (
        f"Target count {len(target)} "
        f"!= morphology count "
        f"{len(morphology)}"
    )

    print(
        f"\nSource shape: "
        f"{source.shape}"
    )

    print(
        f"Target shape: "
        f"{target.shape}"
    )

    print(
        f"Morphology sentences: "
        f"{len(morphology):,}"
    )

    # --------------------------------------------------------
    # Create Dataset
    # --------------------------------------------------------

    dataset = MorphologyTranslationDataset(
        source,
        target,
        morphology
    )

    print(
        f"\nDataset size: "
        f"{len(dataset):,}"
    )

    return dataset


# ============================================================
# TEST DATASET
# ============================================================

if __name__ == "__main__":

    dataset = create_dataset()

    print("\nTesting first sample...")

    sample = dataset[0]

    print(
        "Source shape:",
        sample["source"].shape
    )

    print(
        "Target shape:",
        sample["target"].shape
    )

    print(
        "Morphology shape:",
        sample["morphology"].shape
    )

    print(
        "\nSource first 20 tokens:"
    )

    print(
        sample["source"][:20]
    )

    print(
        "\nTarget first 20 tokens:"
    )

    print(
        sample["target"][:20]
    )

    print(
        "\nMorphology IDs:"
    )

    print(
        sample["morphology"]
    )

    # --------------------------------------------------------
    # DataLoader test
    # --------------------------------------------------------

    print(
        "\nTesting DataLoader..."
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=morphology_collate
    )

    batch = next(iter(loader))

    print(
        "\nBatch source shape:",
        batch["source"].shape
    )

    print(
        "Batch target shape:",
        batch["target"].shape
    )

    print(
        "Batch morphology shape:",
        batch["morphology"].shape
    )

    print("\nDataLoader test successful.")

    print("=" * 70)