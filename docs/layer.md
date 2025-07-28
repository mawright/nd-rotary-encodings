# RoPEEncodingND

## Overview

`RoPEEncodingND` is the main user-facing API. It is intended to be used in Transformer-type attention as an additional step between the initial Q/K/V in-projection and before the query-key product.

Basic usage example for self-attention:

```python
from torch import nn
from nd_rotary_encodings import RoPEEncodingND


class RoPEAttention_nd(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        use_checkpointing: bool = False,
        forward_only: bool = False,
        inplace: bool = False,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)

        self.pos_encoding = RoPEEncodingND(
            position_dim=2,
            embed_dim=embed_dim,
            n_heads=num_heads,
            rope_base_theta=rope_theta,
            use_checkpointing=use_checkpointing,
            forward_only=forward_only,
            inplace=inplace,
        )

        self.dropout_p = dropout
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, x_positions: Tensor):
        batch_size, seq_len, embed_dim = x.shape
        head_dim = embed_dim // self.num_heads
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q, k = self.pos_encoding(q, x_positions, k)

        q = q.reshape(batch_size, seq_len, self.num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_heads, head_dim).permute(0, 2, 1, 3)

        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p)

        x = x.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
```

---

::: position_encoding_layer.rope_encoding_layer.RoPEEncodingND
    options:
        members:
            - forward
            - position_grid
        show_root_heading: true
        show_root_toc_entry: true
        show_root_full_path: false

---

## Utilities

::: position_encoding_layer.utils
    options:
        members:
            - prep_multilevel_positions
            - get_multilevel_freq_group_pattern
        show_root_heading: false
        show_root_toc_entry: false
        show_root_full_path: false
        heading_level: 3