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

from typing import Tuple

import torch
import torch.nn as nn


class LSTM(nn.Module):
    """LSTM backbone with cache support for streaming inference.
    
    This wrapper around nn.LSTM supports the cache interface required
    by the KWS model for streaming inference.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        
        # Cache size: (num_layers * num_directions, batch_size, hidden_size)
        # For bidirectional: num_directions = 2, else 1
        self.num_directions = 2 if bidirectional else 1
        self.cache_size = (self.num_layers * self.num_directions, hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        in_cache: torch.Tensor = torch.zeros(0, 0, 0, dtype=torch.float)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor (B, T, D)
            in_cache (torch.Tensor): Hidden state cache
                Shape: (num_layers * num_directions, B, hidden_size) for h and c
                If empty, initialize with zeros
        
        Returns:
            torch.Tensor: Output (B, T, hidden_size * num_directions)
            torch.Tensor: Output cache (num_layers * num_directions, B, hidden_size)
        """
        batch_size = x.size(0)
        
        # Initialize hidden states if cache is empty
        if in_cache.size(0) == 0:
            device = x.device
            dtype = x.dtype
            h0 = torch.zeros(
                self.num_layers * self.num_directions,
                batch_size,
                self.hidden_size,
                device=device,
                dtype=dtype
            )
            c0 = torch.zeros(
                self.num_layers * self.num_directions,
                batch_size,
                self.hidden_size,
                device=device,
                dtype=dtype
            )
        else:
            # in_cache shape: (2, num_layers * num_directions, B, hidden_size)
            # First element is h, second is c
            h0 = in_cache[0]  # (num_layers * num_directions, B, hidden_size)
            c0 = in_cache[1]  # (num_layers * num_directions, B, hidden_size)
        
        # Forward through LSTM
        output, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Prepare output cache: stack h and c
        # Shape: (2, num_layers * num_directions, B, hidden_size)
        out_cache = torch.stack([hn, cn], dim=0)
        
        return output, out_cache

    def fuse_modules(self):
        """Placeholder for module fusion (not applicable to LSTM)."""
        pass

