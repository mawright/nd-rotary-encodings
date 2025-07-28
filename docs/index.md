# nd-rotary-encodings

Welcome to the documentation for nd-rotary-encodings.

This repository grew out of an effort at [Berkeley Lab](https://www.lbl.gov) towards applying modern computer vision techniques like Detection Transformers to large, sparse scientific images.
Since scientific images like electron microscope iamges can be very large with very small objects, a key requirement was to be able to process high-resolution images without rescaling them to standard sizes (e.g., a few hundred pixels in width and height) typically used for computer vision.

The `nd-rotary-encodings` repository features highly-optimized PyTorch kernels and layers for RoPE-encoding N-dimensional sequences, with exceptional scaling improvements over standard implementations.