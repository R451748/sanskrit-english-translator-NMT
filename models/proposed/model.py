import math
import torch
import torch.nn as nn


# ============================================================
# POSITIONAL ENCODING
# ============================================================

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        max_len=512,
        dropout=0.1
    ):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        position = torch.arange(
            max_len
        ).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2
            ).float()
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(
            max_len,
            d_model
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        pe = pe.unsqueeze(0)

        self.register_buffer(
            "pe",
            pe
        )

    def forward(self, x):

        x = x + self.pe[:, :x.size(1)]

        return self.dropout(x)


# ============================================================
# MORPHOLOGY ENCODER
# ============================================================

class MorphologyEncoder(nn.Module):

    def __init__(
        self,
        vocab_size=228,
        morphology_embedding_dim=64,
        d_model=256,
        pad_idx=0
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=morphology_embedding_dim,
            padding_idx=pad_idx
        )

        self.projection = nn.Sequential(
            nn.Linear(
                morphology_embedding_dim,
                d_model
            ),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, morphology):

        # morphology:
        # [batch, morphology_length]

        embeddings = self.embedding(
            morphology
        )

        # [batch, length]
        mask = (
            morphology != 0
        ).float()

        # [batch, length, 1]
        mask = mask.unsqueeze(-1)

        # Remove PAD representations
        embeddings = embeddings * mask

        # Number of valid morphology tokens
        lengths = mask.sum(
            dim=1
        ).clamp(min=1)

        # Mean pooling
        pooled = (
            embeddings.sum(dim=1)
            / lengths
        )

        # [batch, d_model]
        morphology_vector = self.projection(
            pooled
        )

        return morphology_vector


# ============================================================
# MORPHOLOGY-AWARE TRANSFORMER
# ============================================================

class MorphologyAwareTransformer(nn.Module):

    def __init__(
        self,

        src_vocab_size,
        tgt_vocab_size,

        morphology_vocab_size=228,

        d_model=256,
        nhead=4,

        num_encoder_layers=3,
        num_decoder_layers=3,

        dim_feedforward=1024,

        dropout=0.1,

        src_pad_idx=0,
        tgt_pad_idx=0,

        max_src_len=128,
        max_tgt_len=256
    ):

        super().__init__()

        self.d_model = d_model

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

        # ----------------------------------------------------
        # Sanskrit embedding
        # ----------------------------------------------------

        self.src_embedding = nn.Embedding(
            src_vocab_size,
            d_model,
            padding_idx=src_pad_idx
        )

        # ----------------------------------------------------
        # English embedding
        # ----------------------------------------------------

        self.tgt_embedding = nn.Embedding(
            tgt_vocab_size,
            d_model,
            padding_idx=tgt_pad_idx
        )

        # ----------------------------------------------------
        # Positional encoding
        # ----------------------------------------------------

        self.src_positional = PositionalEncoding(
            d_model=d_model,
            max_len=max_src_len,
            dropout=dropout
        )

        self.tgt_positional = PositionalEncoding(
            d_model=d_model,
            max_len=max_tgt_len,
            dropout=dropout
        )

        # ----------------------------------------------------
        # Morphology encoder
        # ----------------------------------------------------

        self.morphology_encoder = MorphologyEncoder(
            vocab_size=morphology_vocab_size,
            morphology_embedding_dim=64,
            d_model=d_model,
            pad_idx=0
        )

        # ----------------------------------------------------
        # Transformer
        # ----------------------------------------------------

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,

            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,

            dim_feedforward=dim_feedforward,

            dropout=dropout,

            batch_first=True
        )

        # ----------------------------------------------------
        # Output layer
        # ----------------------------------------------------

        self.output_projection = nn.Linear(
            d_model,
            tgt_vocab_size
        )

        # ----------------------------------------------------
        # Weight initialization
        # ----------------------------------------------------

        self._initialize_weights()

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def _initialize_weights(self):

        for module in self.modules():

            if isinstance(
                module,
                nn.Linear
            ):

                nn.init.xavier_uniform_(
                    module.weight
                )

                if module.bias is not None:

                    nn.init.zeros_(
                        module.bias
                    )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        src,
        morphology,
        tgt
    ):

        # ----------------------------------------------------
        # Source embedding
        # ----------------------------------------------------

        src_emb = self.src_embedding(
            src
        ) * math.sqrt(
            self.d_model
        )

        src_emb = self.src_positional(
            src_emb
        )

        # ----------------------------------------------------
        # Morphology representation
        # ----------------------------------------------------

        morphology_vector = (
            self.morphology_encoder(
                morphology
            )
        )

        # [batch, d_model]
        # → [batch, 1, d_model]

        morphology_vector = (
            morphology_vector
            .unsqueeze(1)
        )

        # ----------------------------------------------------
        # Fuse morphology with every
        # Sanskrit encoder token
        # ----------------------------------------------------

        src_emb = (
            src_emb
            + morphology_vector
        )

        # ----------------------------------------------------
        # Target embedding
        # ----------------------------------------------------

        tgt_emb = self.tgt_embedding(
            tgt
        ) * math.sqrt(
            self.d_model
        )

        tgt_emb = self.tgt_positional(
            tgt_emb
        )

        # ----------------------------------------------------
        # Padding masks
        # ----------------------------------------------------

        src_padding_mask = (
            src == self.src_pad_idx
        )

        tgt_padding_mask = (
            tgt == self.tgt_pad_idx
        )

        # ----------------------------------------------------
        # Causal target mask
        # ----------------------------------------------------

        tgt_len = tgt.size(1)

        tgt_mask = torch.triu(
            torch.ones(
                tgt_len,
                tgt_len,
                device=tgt.device,
                dtype=torch.bool
            ),
            diagonal=1
        )

        # ----------------------------------------------------
        # Transformer
        # ----------------------------------------------------

        output = self.transformer(
            src_emb,
            tgt_emb,

            tgt_mask=tgt_mask,

            src_key_padding_mask=(
                src_padding_mask
            ),

            tgt_key_padding_mask=(
                tgt_padding_mask
            ),

            memory_key_padding_mask=(
                src_padding_mask
            )
        )

        # ----------------------------------------------------
        # Vocabulary prediction
        # ----------------------------------------------------

        output = self.output_projection(
            output
        )

        return output


# ============================================================
# MODEL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("TESTING MORPHOLOGY-AWARE TRANSFORMER")
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\nDevice:",
        device
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    SRC_VOCAB_SIZE = 16000
    TGT_VOCAB_SIZE = 16000
    MORPHOLOGY_VOCAB_SIZE = 228

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = MorphologyAwareTransformer(
        src_vocab_size=SRC_VOCAB_SIZE,
        tgt_vocab_size=TGT_VOCAB_SIZE,
        morphology_vocab_size=MORPHOLOGY_VOCAB_SIZE
    ).to(device)

    # --------------------------------------------------------
    # Dummy inputs
    # --------------------------------------------------------

    batch_size = 2

    src = torch.randint(
        1,
        SRC_VOCAB_SIZE,
        (batch_size, 20),
        device=device
    )

    morphology = torch.randint(
        2,
        MORPHOLOGY_VOCAB_SIZE,
        (batch_size, 8),
        device=device
    )

    tgt = torch.randint(
        1,
        TGT_VOCAB_SIZE,
        (batch_size, 30),
        device=device
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            src,
            morphology,
            tgt
        )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print(
        "\nInput Sanskrit shape:",
        src.shape
    )

    print(
        "Input morphology shape:",
        morphology.shape
    )

    print(
        "Input English shape:",
        tgt.shape
    )

    print(
        "Output shape:",
        output.shape
    )

    # --------------------------------------------------------
    # Parameter count
    # --------------------------------------------------------

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "\nTotal parameters:",
        f"{total_parameters:,}"
    )

    print(
        "Trainable parameters:",
        f"{trainable_parameters:,}"
    )

    print(
        "\nModel forward pass successful."
    )

    print("=" * 70)