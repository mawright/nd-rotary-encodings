# nd-rotary-encodings

[![Tests](https://github.com/mawright/nd-rotary-encodings/actions/workflows/tests.yml/badge.svg)](https://github.com/mawright/nd-rotary-encodings/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/mawright/nd-rotary-encodings/branch/main/graph/badge.svg)](https://codecov.io/gh/mawright/nd-rotary-encodings)
![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)
[![License](https://img.shields.io/github/license/mawright/nd-rotary-encodings)](https://github.com/mawright/nd-rotary-encodings/blob/main/LICENSE)

N-Dimensional Rotary Positional Encodings (RoPE) in PyTorch

## Introduction

Rotary Positional Encodings (RoPE) are the modern method to encode positional information into Transformer inputs.
While RoPE is most well-known for its use in 1D text and timeseries data, recent research has extended it to multidimensional data like 2D images and 3D volumes.
This repository contains an implementation of N-dimensional RoPE that builds on and extends prior work with enhanced performance and new features.

## Background

[Heo et al.](https://arxiv.org/abs/2403.13298) of NAVER AI Lab proposed an [extension of RoPE to 2D for Vision Transformers](https://github.com/naver-ai/rope-vit).
Called RoPE-Mixed, the formulation extends traditional 1D RoPE by extending the RoPE rotation matrix $\mathbf{R} \in \mathbb{C}^{N \times (d_{head} / 2)}$
```math
\mathbf{R}_{[n, t]} = e^{i \theta_t n}
```
where $\mathbf{R_{[n,t]}}$ means the $[n, t]$ element of $\mathbf{R}$, $n \in \{1, \dots, N\}$ indexes the 1D spatial position, $t \in \{1, \dots, d_{head}/2\}$ indexes the feature dimension, $\theta_t$ is a learned or precomputed scalar frequency, and $i=\sqrt{-1}$; to a form where each entry of $\mathbf{R}$ has equal contributions from both spatial axes $x$ and $y$:
```math
\mathbf{R}_{[n, t]} = e^{i (\theta_t^x p_n^x + \theta_t^y p_n^y)}
```
where now $\theta_t^x$ and $\theta_t^y$ are separate per-axis scalar frequencies, and $p_n^x$ and $p_n^y$ are the 2D spatial coordinates.

This repository is the result of research into further generalizing RoPE-Mixed to higher dimensions, with a particular focus on small object detection using DETRs on very large and sparse images.

## Implementation Features

Compared to the [official implementation](https://github.com/naver-ai/rope-vit) of RoPE-Mixed, our implementation offers:

- Improved performance (see benchmarks below)
- Generalization to ND spaces with N > 2
- Support for arbitrary, non-grid positions (for representing, e.g., arbitrary object positions)
- Gradient support for position tensors
- [Implementation documentation](https://mawright.github.io/nd-rotary-encodings/)
- Comprehensive unit tests and property-based tests using [Hypothesis](hypothesis.readthedocs.io/)
- Encoder-decoder attention (a.k.a. cross-attention) support
- Experimental "grouped dimensions" construction for application to network modules beyond ViT backbones, such as detection transformers (DETRs)
- Custom gradient checkpointing logic for memory-constrained contexts (e.g., in DETR-like encoder-decoder setups where the number of key/value embeddings corresponding to pixel features may be much larger than the number of object queries)

## Benchmarks

The benchmarks below show performance of a single Multi-head Attention (MHA) layer with no Rotary Positional Encodings (i.e., vanilla MHA), an MHA layer with the reference RoPE-Mixed implementation applied to the embeddings immediately before the query-key product, and an MHA layer with our nd-RoPE implementation in the same place.
The test data are random square images of size NxN, with batch size 4, embedding dimension 256, and 8 heads. The benchmarks were run in float32 precision on a single A100 GPU. The benchmark may be reproduced by running the [benchmark notebook](notebooks/benchmark.ipynb).

The image sizes along the x axes of the benchmark results denote the actual token counts of the processed embeddings without the typical downsampling/patchification used in most Vision Transformers.

### Forward pass

<img src="images/memory_forward.png" width="500"/> <img src="images/walltime_forward.png" width="500"/>

On the forward pass, our implementation achieves significantly better memory scaling than the reference implementation, bringing the memory scaling from an apparent high-degree polynomial scaling increase to a constant multiplicative increase above the no-RoPE layer in training mode.
In inference mode, additional optimizations such as in-place rotation of the embeddings allow us to bring the marginal memory cost to virtually no marginal increase.
Importantly, our implementation allows token counts beyond a few thousand (e.g., 64x64) to be processed by a RoPE-enabled ViT layer or similar.

Forward-pass runtime is also significantly improved, with our implementation adding a small increase in walltime that becomes relatively insignificant past ~48x48.

### Backward pass

<img src="images/memory_backward.png" width="500"/> <img src="images/walltime_backward.png" width="500"/>

Similar to the forward pass, our implementation brings marginal memory consumption from apparent high-degree polynomial scaling to a constant multiplicative increase above the No-RoPE case.

The trends in walltime are also similar, with the reference implementation adding a marginally large walltime increase and ours adding a small factor that washes out at moderate to high resolutions.

## Installation

`nd-rotary-encodings` has no requirements beyond base PyTorch.
To install, simply clone the repository and use pip:

```bash
git clone https://github.com/mawright/nd-rotary-encodings
cd nd-rotary-encodings
pip install -e .  # editable installation
```

To run the test suite, you'll need to install the optional dependencies (pytest and Hypothesis):

```bash
pip install -e ".[tests]"
```

## API

The high-level user-facing interface is the `nn.Module` [`RoPEEncodingND`](https://mawright.github.io/nd-rotary-encodings/layer/#position_encoding_layer.rope_encoding_layer.RoPEEncodingND).
This layer takes the query embedding tensor, the query position tensor, and optionally the key and key position tensor, and returns RoPE-encoded versions of the query and key tensors.

A few usage examples:

- Basic 3D RoPE encoding of queries for self-attention:

```python
import torch
from nd_rotary_encodings import RoPEEncodingND

# Architecture parameters
position_dim = 3
embed_dim = 128
n_heads = 4

# Query tensor parameters
batch_size = 4
seq_length = 16
embed_dim = 128

# Create a RoPE layer for 3D positions with embedding dimension of 128 and 4 heads
rope = RoPEEncodingND(position_dim, embed_dim, n_heads)

# Create query tensor and corresponding positions
query = torch.randn(batch_size, seq_length, embed_dim)
query_pos = torch.randn(batch_size, seq_length, position_dim)  # float positions supported

rotated_query = rope(query, query_pos)

assert not torch.allclose(query, rotated_query)
```

- The same layer can be used for encoder-decoder attention with both a query and key tensor:

```python
key_seq_length = 32

key = torch.randn(batch_size, key_seq_length, embed_dim)
key_pos = torch.randn(batch_size, key_seq_length, position_dim)

rotated_query_2, rotated_key = rope(query, query_pos, key, key_pos)

assert torch.equal(rotated_query, rotated_query_2)
assert not torch.allclose(key, rotated_key)
```

For more information on usage, see the [documentation page for `RoPEEncodingND`](https://mawright.github.io/nd-rotary-encodings/layer/).

## See Also

- [pytorch-sparse-utils](https://github.com/mawright/pytorch-sparse-utils): Low-level utilities for dealing with large, sparse tensors.
- [sparse-transformer-layers](https://github.com/mawright/sparse-transformer-layers): Implementations of Transformer layers built on this repository's RoPE encoding layer tailored to sparse tensors, including variants like Multi-scale Deformable Attention.

## Future Plans

- Integration of [LieRE](https://github.com/Stanford-AIMI/LieRE), [STRING](https://arxiv.org/abs/2502.02562), and/or other more-advanced schemes for RoPE rotation matrices
- Additional benchmarks
- Expanded usage examples for the more advanced and experimental features
