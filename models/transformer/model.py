import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Adds positional information to token embeddings.
    """

    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2)
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x shape:
        [batch_size, sequence_length, d_model]
        """

        x = x + self.pe[:, :x.size(1)]

        return self.dropout(x)


class SanskritEnglishTransformer(nn.Module):
    """
    Transformer Encoder-Decoder model
    for Sanskrit -> English translation.
    """

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
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

        # Sanskrit embedding
        self.src_embedding = nn.Embedding(
            src_vocab_size,
            d_model,
            padding_idx=src_pad_idx
        )

        # English embedding
        self.tgt_embedding = nn.Embedding(
            tgt_vocab_size,
            d_model,
            padding_idx=tgt_pad_idx
        )

        # Positional encodings
        self.src_positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=max_src_len,
            dropout=dropout
        )

        self.tgt_positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=max_tgt_len,
            dropout=dropout
        )

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # Output layer
        self.output_layer = nn.Linear(
            d_model,
            tgt_vocab_size
        )

        self._init_weights()

    def _init_weights(self):
        """
        Xavier initialization for linear layers.
        """

        for module in self.modules():

            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def generate_square_subsequent_mask(self, size, device):
        """
        Creates causal mask for the decoder.

        This prevents the decoder from seeing
        future target tokens.
        """

        mask = torch.triu(
            torch.ones(
                size,
                size,
                device=device,
                dtype=torch.bool
            ),
            diagonal=1
        )

        return mask

    def forward(self, src, tgt):
        """
        Forward pass.

        src:
            [batch_size, source_length]

        tgt:
            [batch_size, target_length]

        Returns:
            logits:
            [batch_size, target_length, tgt_vocab_size]
        """

        # Padding masks
        src_padding_mask = src.eq(self.src_pad_idx)

        tgt_padding_mask = tgt.eq(self.tgt_pad_idx)

        # Causal decoder mask
        tgt_mask = self.generate_square_subsequent_mask(
            tgt.size(1),
            tgt.device
        )

        # Embeddings
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)

        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)

        # Positional encoding
        src_emb = self.src_positional_encoding(src_emb)

        tgt_emb = self.tgt_positional_encoding(tgt_emb)

        # Transformer
        output = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )

        # Vocabulary prediction
        logits = self.output_layer(output)

        return logits


if __name__ == "__main__":

    # Small test to verify that the model works.

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("TRANSFORMER MODEL TEST")
    print("=" * 60)

    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # Example vocabulary sizes.
    # Actual vocabulary sizes will come from SentencePiece.
    SRC_VOCAB_SIZE = 16000
    TGT_VOCAB_SIZE = 16000

    model = SanskritEnglishTransformer(
        src_vocab_size=SRC_VOCAB_SIZE,
        tgt_vocab_size=TGT_VOCAB_SIZE
    ).to(device)

    # Dummy input
    batch_size = 2
    src_length = 128
    tgt_length = 256

    src = torch.randint(
        0,
        SRC_VOCAB_SIZE,
        (batch_size, src_length),
        device=device
    )

    tgt = torch.randint(
        0,
        TGT_VOCAB_SIZE,
        (batch_size, tgt_length),
        device=device
    )

    # Forward pass
    with torch.no_grad():
        output = model(src, tgt)

    print("Input Sanskrit shape:", src.shape)
    print("Input English shape:", tgt.shape)
    print("Output shape:", output.shape)

    print("\nExpected output:")
    print(
        f"[{batch_size}, {tgt_length}, {TGT_VOCAB_SIZE}]"
    )

    print("\nModel parameters:")

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\nModel test completed successfully!")