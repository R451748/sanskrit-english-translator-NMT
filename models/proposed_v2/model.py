import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=256, dropout=0.1):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(
            0,
            max_len,
            dtype=torch.float
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
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

        x = x + self.pe[:, :x.size(1)]

        return self.dropout(x)


class MorphologyConditionedTransformer(nn.Module):

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        morphology_vocab_size,
        d_model=256,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=1024,
        dropout=0.1,
        src_pad_idx=0,
        tgt_pad_idx=0,
        morph_pad_idx=0,
        max_src_len=128,
        max_morph_len=64,
        max_tgt_len=256
    ):

        super().__init__()

        self.d_model = d_model

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.morph_pad_idx = morph_pad_idx

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

        # Morphology tag embedding
        self.morph_embedding = nn.Embedding(
            morphology_vocab_size,
            d_model,
            padding_idx=morph_pad_idx
        )

        # Segment embeddings
        # 0 = morphology
        # 1 = Sanskrit
        self.segment_embedding = nn.Embedding(
            2,
            d_model
        )

        self.encoder_positional = PositionalEncoding(
            d_model,
            max_len=max_src_len + max_morph_len,
            dropout=dropout
        )

        self.decoder_positional = PositionalEncoding(
            d_model,
            max_len=max_tgt_len,
            dropout=dropout
        )

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.output_layer = nn.Linear(
            d_model,
            tgt_vocab_size
        )

        self._init_weights()

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
                    mean=0.0,
                    std=0.02
                )

        with torch.no_grad():

            self.src_embedding.weight[
                self.src_pad_idx
            ].zero_()

            self.tgt_embedding.weight[
                self.tgt_pad_idx
            ].zero_()

            self.morph_embedding.weight[
                self.morph_pad_idx
            ].zero_()

    def generate_causal_mask(
        self,
        size,
        device
    ):

        return torch.triu(
            torch.ones(
                size,
                size,
                dtype=torch.bool,
                device=device
            ),
            diagonal=1
        )

    def forward(
        self,
        src,
        morphology,
        tgt
    ):

        # ============================================
        # Sanskrit embeddings
        # ============================================

        src_emb = self.src_embedding(src)

        src_emb = src_emb * math.sqrt(
            self.d_model
        )

        # ============================================
        # Morphology embeddings
        # ============================================

        morph_emb = self.morph_embedding(
            morphology
        )

        morph_emb = morph_emb * math.sqrt(
            self.d_model
        )

        # ============================================
        # Segment embeddings
        # ============================================

        morph_segments = torch.zeros(
            morphology.shape,
            dtype=torch.long,
            device=morphology.device
        )

        src_segments = torch.ones(
            src.shape,
            dtype=torch.long,
            device=src.device
        )

        morph_emb = (
            morph_emb
            + self.segment_embedding(
                morph_segments
            )
        )

        src_emb = (
            src_emb
            + self.segment_embedding(
                src_segments
            )
        )

        # ============================================
        # Combine morphology + Sanskrit
        # ============================================

        encoder_input = torch.cat(
            [
                morph_emb,
                src_emb
            ],
            dim=1
        )

        # ============================================
        # Padding masks
        # ============================================

        morph_mask = (
            morphology == self.morph_pad_idx
        )

        src_mask = (
            src == self.src_pad_idx
        )

        encoder_padding_mask = torch.cat(
            [
                morph_mask,
                src_mask
            ],
            dim=1
        )

        # ============================================
        # Encoder positional encoding
        # ============================================

        encoder_input = self.encoder_positional(
            encoder_input
        )

        # ============================================
        # Target embedding
        # ============================================

        tgt_emb = self.tgt_embedding(tgt)

        tgt_emb = tgt_emb * math.sqrt(
            self.d_model
        )

        tgt_emb = self.decoder_positional(
            tgt_emb
        )

        # ============================================
        # Target padding mask
        # ============================================

        tgt_padding_mask = (
            tgt == self.tgt_pad_idx
        )

        # ============================================
        # Causal decoder mask
        # ============================================

        tgt_mask = self.generate_causal_mask(
            tgt.size(1),
            tgt.device
        )

        # ============================================
        # Transformer
        # ============================================

        output = self.transformer(
            src=encoder_input,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=encoder_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=encoder_padding_mask
        )

        # ============================================
        # Output vocabulary
        # ============================================

        output = self.output_layer(output)

        return output