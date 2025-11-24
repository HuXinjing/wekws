#!/usr/bin/env python3
# Copyright (c) 2021 Binbin Zhang
#               2023 Jing Du
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape (seq_len, batch_size, d_model)
        """
        x = x + self.pe[:x.size(0), :]
        return x


class StreamingTransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with streaming support using causal attention."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            src: (B, T, D)
            src_mask: Causal mask for streaming (T, T)
            src_key_padding_mask: (B, T)
        """
        # Self-attention with causal mask
        src2 = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feedforward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class StreamingTransformer(nn.Module):
    """Streaming Transformer encoder with cache support.
    
    Uses causal (masked) self-attention to support streaming inference.
    Cache stores the key-value pairs from previous frames.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        max_cache_len: int = 1000,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.max_cache_len = max_cache_len
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_cache_len)
        
        # Transformer layers
        encoder_layers = [
            StreamingTransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout, activation
            )
            for _ in range(num_layers)
        ]
        self.layers = nn.ModuleList(encoder_layers)
        
        # Layer norm
        self.norm = nn.LayerNorm(d_model)

    def _generate_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """Generate causal mask for streaming attention."""
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(
        self,
        x: torch.Tensor,
        in_cache: torch.Tensor = torch.zeros(0, 0, 0, dtype=torch.float)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor (B, T, D)
            in_cache (torch.Tensor): Key-value cache from previous frames
                Shape: (num_layers, 2, B, nhead, cache_len, d_k)
                If empty, no cache is used
        
        Returns:
            torch.Tensor: Output (B, T, D)
            torch.Tensor: Output cache (num_layers, 2, B, nhead, new_cache_len, d_k)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Add positional encoding
        # Convert to (T, B, D) for positional encoding
        x = x.transpose(0, 1)  # (T, B, D)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (B, T, D)
        
        # Generate causal mask for current sequence
        causal_mask = self._generate_causal_mask(seq_len, device)
        
        # Process through transformer layers
        # For simplicity, we use standard attention without KV cache
        # In production, you might want to implement KV cache for better efficiency
        for layer in self.layers:
            x = layer(x, src_mask=causal_mask)
        
        x = self.norm(x)
        
        # For now, return empty cache (can be extended to support KV cache)
        # Shape: (num_layers, 2, B, nhead, 0, d_k)
        d_k = self.d_model // self.nhead
        out_cache = torch.zeros(
            self.num_layers, 2, batch_size, self.nhead, 0, d_k,
            device=device, dtype=x.dtype
        )
        
        return x, out_cache

    def fuse_modules(self):
        """Placeholder for module fusion."""
        pass

