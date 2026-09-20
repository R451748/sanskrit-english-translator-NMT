import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


SEQUENCE_DIR = "dataset/processed/sequences"

PAD_IDX = 0


class TranslationDataset(Dataset):
    """
    Sanskrit -> English translation dataset.

    Loads the already tokenized .npy sequences.
    """

    def __init__(self, split):

        source_file = os.path.join(
            SEQUENCE_DIR,
            f"{split}_source.npy"
        )

        target_file = os.path.join(
            SEQUENCE_DIR,
            f"{split}_target.npy"
        )

        if not os.path.exists(source_file):
            raise FileNotFoundError(
                f"Source file not found: {source_file}"
            )

        if not os.path.exists(target_file):
            raise FileNotFoundError(
                f"Target file not found: {target_file}"
            )

        self.source = np.load(source_file)
        self.target = np.load(target_file)

        if len(self.source) != len(self.target):
            raise ValueError(
                "Source and target sizes do not match."
            )

        print(f"{split.capitalize()} dataset loaded")
        print(f"Source shape: {self.source.shape}")
        print(f"Target shape: {self.target.shape}")

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):

        source = torch.tensor(
            self.source[index],
            dtype=torch.long
        )

        target = torch.tensor(
            self.target[index],
            dtype=torch.long
        )

        return source, target


def dynamic_collate(batch):
    """
    Dynamically pads each batch.

    Instead of always using:
        source = 128
        target = 256

    it finds the longest sequence in the current batch.
    """

    sources, targets = zip(*batch)

    # --------------------------------------------------------
    # Find actual lengths
    # --------------------------------------------------------

    source_lengths = [
        (source != PAD_IDX).sum().item()
        for source in sources
    ]

    target_lengths = [
        (target != PAD_IDX).sum().item()
        for target in targets
    ]

    max_source_length = max(source_lengths)

    max_target_length = max(target_lengths)

    # --------------------------------------------------------
    # Create padded tensors
    # --------------------------------------------------------

    source_batch = torch.full(
        (
            len(sources),
            max_source_length
        ),
        PAD_IDX,
        dtype=torch.long
    )

    target_batch = torch.full(
        (
            len(targets),
            max_target_length
        ),
        PAD_IDX,
        dtype=torch.long
    )

    # --------------------------------------------------------
    # Copy actual tokens
    # --------------------------------------------------------

    for i, source in enumerate(sources):

        length = source_lengths[i]

        source_batch[i, :length] = source[:length]

    for i, target in enumerate(targets):

        length = target_lengths[i]

        target_batch[i, :length] = target[:length]

    return source_batch, target_batch


def create_dataloader(
    split,
    batch_size=8,
    shuffle=False,
    num_workers=0
):

    dataset = TranslationDataset(split)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=dynamic_collate,
        pin_memory=torch.cuda.is_available()
    )

    return loader


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("DYNAMIC PADDING DATASET TEST")
    print("=" * 70)

    loader = create_dataloader(
        split="train",
        batch_size=8,
        shuffle=True
    )

    source, target = next(iter(loader))

    print("\nBatch loaded successfully!")

    print("Source batch shape:", source.shape)
    print("Target batch shape:", target.shape)

    print("\nSource padding percentage:")

    source_padding = (
        (source == PAD_IDX).float().mean() * 100
    )

    print(f"{source_padding:.2f}%")

    print("\nTarget padding percentage:")

    target_padding = (
        (target == PAD_IDX).float().mean() * 100
    )

    print(f"{target_padding:.2f}%")

    print("\nSource sequence lengths:")

    source_lengths = (
        source != PAD_IDX
    ).sum(dim=1)

    print(source_lengths.tolist())

    print("\nTarget sequence lengths:")

    target_lengths = (
        target != PAD_IDX
    ).sum(dim=1)

    print(target_lengths.tolist())

    print("\nDevice:", source.device)

    print("\n" + "=" * 70)
    print("DYNAMIC PADDING TEST COMPLETED")
    print("=" * 70)