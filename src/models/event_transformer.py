from __future__ import annotations

import torch
import torch.nn as nn


class EventTransformer(nn.Module):
    """
    Transformer encoder that embeds:
      - variable_id
      - value_bin
      - dt_bucket
      - position
    and predicts variable_id + value_bin for masked events.
    """

    def __init__(
        self,
        n_var_tokens: int,
        n_val_tokens: int,
        n_dt_buckets: int = 16,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()

        self.n_var_tokens = n_var_tokens
        self.n_val_tokens = n_val_tokens
        self.n_dt_buckets = n_dt_buckets
        self.d_model = d_model
        self.max_len = max_len

        self.var_emb = nn.Embedding(n_var_tokens, d_model)
        self.val_emb = nn.Embedding(n_val_tokens, d_model)
        self.dt_emb = nn.Embedding(n_dt_buckets, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(d_model)
        self.var_head = nn.Linear(d_model, n_var_tokens)
        self.val_head = nn.Linear(d_model, n_val_tokens)

    def forward(self, var_ids, val_ids, dt_bucket, attn_mask=None):
        """
        var_ids:   (B, T)
        val_ids:   (B, T)
        dt_bucket: (B, T)
        attn_mask: (B, T) bool, True for real tokens, False for padding
        """
        B, T = var_ids.shape
        if T > self.max_len:
            raise ValueError(f"Sequence length {T} exceeds max_len={self.max_len}")

        pos = torch.arange(T, device=var_ids.device).unsqueeze(0).expand(B, T)

        x = self.var_emb(var_ids) + self.val_emb(val_ids) + self.dt_emb(dt_bucket) + self.pos_emb(pos)

        # Transformer expects key_padding_mask=True for pads. Our attn_mask is True for real tokens.
        key_padding_mask = None
        if attn_mask is not None:
            key_padding_mask = ~attn_mask  # pads = True

        h = self.encoder(x, src_key_padding_mask=key_padding_mask)
        h = self.norm(h)

        var_logits = self.var_head(h)  # (B,T,n_var_tokens)
        val_logits = self.val_head(h)  # (B,T,n_val_tokens)
        return var_logits, val_logits
    def encode(self, var_ids, val_ids, dt_bucket, attn_mask=None):
        """
        Returns hidden states h: (B, T, d_model)
        """
        B, T = var_ids.shape
        if T > self.max_len:
            raise ValueError(f"Sequence length {T} exceeds max_len={self.max_len}")

        pos = torch.arange(T, device=var_ids.device).unsqueeze(0).expand(B, T)

        x = self.var_emb(var_ids) + self.val_emb(val_ids) + self.dt_emb(dt_bucket) + self.pos_emb(pos)

        key_padding_mask = None
        if attn_mask is not None:
            key_padding_mask = ~attn_mask  # pads=True

        h = self.encoder(x, src_key_padding_mask=key_padding_mask)
        h = self.norm(h)
        return h
