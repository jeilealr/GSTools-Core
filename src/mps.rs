//! MPS (Multiple Point Statistics) kernels for Direct Sampling.
//!
//! Fused gather + distance functions that replace the Python `_dist_block`
//! closure in `gstools/mps/scan.py`. Each kernel reads TI candidates via
//! flat-index arithmetic and computes weighted distances in one pass,
//! avoiding the temporary `all_de_ti` allocation of the Python path.
