# Functional API

The functional API contains the low-level operations used by `RoPEEncodingND`.
They include custom autograd Functions, TorchScript-optimized forward and backward kernels, and various wrappers.

The application of RoPE from embeddings, positions, and RoPE frequencies is broken into two steps:

    1. Compute the RoPE rotation tensor from positions and frequencies
    2. Rotate the embeddings by multiplying them by the RoPE rotation tensor.

The RoPE rotation tensor is fast to compute but may potentially be very large if the sequence dimension and/or batch size are large, so allowing it to be recomputed during the backward pass is a key optimization.

---

## Forward and backward kernels

::: functional.forward_backward_fns
    options:
        members:
            - calculate_rope
            - calculate_rope_backward
            - rotate_embeddings
            - rotate_embeddings_backward
        show_root_heading: false
        show_root_toc_entry: false
        show_root_full_path: false
        heading_level: 3

---

## Checkpointed wrappers

::: functional.autograd
    options:
        members:
            - calculate_rope_checkpointed
            - rotate_embeddings_checkpointed
            - apply_rope_checkpointed
        show_root_heading: false
        show_root_toc_entry: false
        show_root_full_path: false
        heading_level: 3

---

## Forward-only wrappers

These wrappers enable additional marginal optimizations suitable when only forward (inference) mode is needed.

::: functional.forward_only
    options:
        members:
            - rotate_embeddings_forward_only
            - apply_rope_forward_only
        show_root_heading: false
        show_root_toc_entry: false
        show_root_full_path: false
        heading_level: 3