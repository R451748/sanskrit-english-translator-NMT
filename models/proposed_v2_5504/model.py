import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=512):

        super().__init__()

        position = torch.arange(
            max_len,
            dtype=torch.float
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
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

        self.register_buffer("pe", pe)

    def forward(self, x):

        return x + self.pe[:, :x.size(1)]


class MorphologyConditionedTransformer(nn.Module):

    def __init__(
        self,
        src_vocab_size=16000,
        tgt_vocab_size=16000,
        morph_vocab_size=392,

        d_model=256,
        nhead=4,

        num_encoder_layers=3,
        num_decoder_layers=3,

        dim_feedforward=1024,
        dropout=0.1,

        src_pad_idx=0,
        tgt_pad_idx=0,
        morph_pad_idx=0
    ):

        super().__init__()

        self.d_model = d_model

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.morph_pad_idx = morph_pad_idx

        # ----------------------------------------------------
        # Embeddings
        # ----------------------------------------------------

        self.src_embedding = nn.Embedding(
            src_vocab_size,
            d_model,
            padding_idx=src_pad_idx
        )

        self.tgt_embedding = nn.Embedding(
            tgt_vocab_size,
            d_model,
            padding_idx=tgt_pad_idx
        )

        self.morph_embedding = nn.Embedding(
            morph_vocab_size,
            d_model,
            padding_idx=morph_pad_idx
        )

        # ----------------------------------------------------
        # Segment embedding
        #
        # 0 = morphology
        # 1 = Sanskrit
        # ----------------------------------------------------

        self.segment_embedding = nn.Embedding(
            2,
            d_model
        )

        # ----------------------------------------------------
        # Positional encoding
        # ----------------------------------------------------

        self.positional_encoding = PositionalEncoding(
            d_model,
            max_len=512
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
        # Output projection
        # ----------------------------------------------------

        self.output_projection = nn.Linear(
            d_model,
            tgt_vocab_size
        )

        self._init_weights()

    # ========================================================
    # Weight initialization
    # ========================================================

    def _init_weights(self):

        for module in self.modules():

            if isinstance(module, nn.Linear):

                nn.init.xavier_uniform_(
                    module.weight
                )

                if module.bias is not None:

                    nn.init.zeros_(
                        module.bias
                    )

            elif isinstance(module, nn.Embedding):

                nn.init.normal_(
                    module.weight,
                    mean=0,
                    std=0.02
                )

                if module.padding_idx is not None:

                    nn.init.zeros_(
                        module.weight[
                            module.padding_idx
                        ]
                    )

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        src,
        morphology,
        tgt
    ):

        # ----------------------------------------------------
        # Source embeddings
        # ----------------------------------------------------

        src_emb = self.src_embedding(src)

        src_emb = src_emb * math.sqrt(
            self.d_model
        )

        # ----------------------------------------------------
        # Morphology embeddings
        # ----------------------------------------------------

        morph_emb = self.morph_embedding(
            morphology
        )

        morph_emb = morph_emb * math.sqrt(
            self.d_model
        )

        # ----------------------------------------------------
        # Segment embeddings
        # ----------------------------------------------------

        morph_len = morphology.size(1)
        src_len = src.size(1)

        morph_segment_ids = torch.zeros(
            morphology.size(),
            dtype=torch.long,
            device=morphology.device
        )

        src_segment_ids = torch.ones(
            src.size(),
            dtype=torch.long,
            device=src.device
        )

        morph_emb = morph_emb + self.segment_embedding(
            morph_segment_ids
        )

        src_emb = src_emb + self.segment_embedding(
            src_segment_ids
        )

        # ----------------------------------------------------
        # Combine morphology + Sanskrit
        # ----------------------------------------------------

        encoder_input = torch.cat(
            [
                morph_emb,
                src_emb
            ],
            dim=1
        )

        encoder_input = self.positional_encoding(
            encoder_input
        )

        # ----------------------------------------------------
        # Encoder padding mask
        # ----------------------------------------------------

        morph_padding_mask = (
            morphology == self.morph_pad_idx
        )

        src_padding_mask = (
            src == self.src_pad_idx
        )

        encoder_padding_mask = torch.cat(
            [
                morph_padding_mask,
                src_padding_mask
            ],
            dim=1
        )

        # ----------------------------------------------------
        # Target
        # ----------------------------------------------------

        tgt_input = tgt[:, :-1]

        tgt_emb = self.tgt_embedding(
            tgt_input
        )

        tgt_emb = tgt_emb * math.sqrt(
            self.d_model
        )

        tgt_emb = self.positional_encoding(
            tgt_emb
        )

        tgt_padding_mask = (
            tgt_input == self.tgt_pad_idx
        )

        # ----------------------------------------------------
        # Causal mask
        # ----------------------------------------------------

        tgt_len = tgt_input.size(1)

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
            src=encoder_input,

            tgt=tgt_emb,

            tgt_mask=tgt_mask,

            src_key_padding_mask=encoder_padding_mask,

            memory_key_padding_mask=encoder_padding_mask,

            tgt_key_padding_mask=tgt_padding_mask
        )

        # ----------------------------------------------------
        # Vocabulary projection
        # ----------------------------------------------------

        output = self.output_projection(
            output
        )

        return output


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("TESTING V2-5504 MORPHOLOGY-CONDITIONED TRANSFORMER")
    print("=" * 70)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("\nDevice:", device)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    model = MorphologyConditionedTransformer(
        src_vocab_size=16000,
        tgt_vocab_size=16000,
        morph_vocab_size=392
    ).to(device)

    # --------------------------------------------------------
    # Dummy input
    # --------------------------------------------------------

    src = torch.randint(
        1,
        16000,
        (2, 20),
        device=device
    )

    morphology = torch.tensor(
        [
            [2, 16, 2, 2, 18, 105, 2, 0],
            [4, 4, 10, 4, 4, 0, 0, 0]
        ],
        dtype=torch.long,
        device=device
    )

    tgt = torch.randint(
        1,
        16000,
        (2, 30),
        device=device
    )

    # Add padding examples
    src[:, -2:] = 0
    tgt[:, -2:] = 0

    with torch.no_grad():

        output = model(
            src,
            morphology,
            tgt
        )

    print("\nInput shapes:")

    print("Source:", src.shape)
    print("Morphology:", morphology.shape)
    print("Target:", tgt.shape)

    print("\nOutput shape:")
    print(output.shape)

    # --------------------------------------------------------
    # Parameter count
    # --------------------------------------------------------

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "\nTotal parameters:",
        f"{total_params:,}"
    )

    print("\n" + "=" * 70)
    print("MODEL TEST COMPLETE")
    print("=" * 70)