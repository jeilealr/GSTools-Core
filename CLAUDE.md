# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## What this repo is

`gstools-core` is the Rust backend of the Python package
[GSTools](../GSTools). It is a drop-in replacement for the Cython backend
(`gstools-cython`). Python auto-selects it at runtime via
`gstools/config.py:USE_GSTOOLS_CORE`. The compiled output is a single Python
extension module (`gstools_core`) over numpy arrays: the original 9 field /
kriging / variogram functions, plus a full MPS Direct Sampling engine.

The MPS surface has grown well beyond the original univariate block kernels
(see `src/mps.rs`). It now owns the complete simulation in Rust via
`src/mps_engine.rs`: multivariate co-simulation, per-variable distance kernels
(L_p, variation, categorical, categorical **penalty matrices**), nonstationary
rotation/scale lag transforms, **zoned** multi-TI simulation, sequential/random
TI scan order, and serial-or-parallel **post-processing (cleanup) passes**. The
Python ABI is versioned: `mps_api_version()` returns the current native MPS
revision (**7** at the time of writing) and the Python layer
(`gstools/mps/_core.py`) gates on it, so a stale same-version Core is rejected
with an actionable rebuild message. Bump `mps_api_version_py` in lockstep with
any change to an MPS entry-point signature.

## Environment

The working Python environment is the `gstools` conda env:

```bash
/Users/lealroja/miniconda3/envs/gstools/bin/python3
```

maturin must be run with CONDA_PREFIX unset (conda and maturin conflict):

```bash
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV maturin develop --release
```

Or use the shared venv that already has maturin and numpy installed:

```bash
/Users/lealroja/Documents/UFZ/RUST/.venv/bin/maturin develop --release
```

## Commands

**Build and install into the active Python environment (development):**
```bash
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV maturin develop
```

**Build release (optimised — use for benchmarking):**
```bash
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV maturin develop --release
```

**Run Rust unit tests:**
```bash
cargo test
```

**Check for warnings (must be zero before committing):**
```bash
cargo clippy --all-targets --all-features -- -D warnings
```

**Format:**
```bash
cargo fmt
```

**Run benchmarks:**
```bash
cargo bench
```

**Verify the Python round-trip after a change:**
```python
# from the gstools conda env, after maturin develop
import gstools_core
import numpy as np
# quick smoke-test — adapt to the function you changed
result = gstools_core.summate(cov_samples, z1, z2, pos, None)
```

**Enable the Rust backend in GSTools:**
```python
import gstools as gs
gs.config.USE_GSTOOLS_CORE = True   # already True if gstools_core is installed
```

## Architecture

Five source files, one private helper:

| File | What it does |
|---|---|
| `src/lib.rs` | PyO3 module entry point. Declares `#[pymodule] gstools_core` and wraps each Rust function as a `#[pyfunction]`. All numpy array marshalling lives here, including `mps_simulate` / `mps_cleanup_pass` (own their inputs before `py.detach()`) and the `mps_api_version` ABI gate. |
| `src/field.rs` | Random field generation (`summator`, `summator_incompr`, `summator_fourier`). Uses const-generic `ShortVec<N>` + rayon for parallelism. |
| `src/krige.rs` | Kriging matrix operations (`calculator_field_krige`, `calculator_field_krige_and_variance`). Parallel fold/reduce over position columns. |
| `src/variogram.rs` | Variogram estimation (structured, masked-structured, directional, unstructured). Trait-based estimator/distance selection via `choose_estimator!` / `choose_distance!` macros. |
| `src/mps.rs` | MPS per-node distance/scan kernels: fused gather + weighted distance over a block of TI candidates (`dist_block_categorical/l1/l2/lp/variation`, masked + rayon variants), the categorical **penalty** cost, and the multivariate node scan `scan_node` (plus the legacy univariate `scan_node_categorical`). The node scan supports sequential and node-local random scan order. |
| `src/mps_engine.rs` | The Rust-owned Direct Sampling engine. `simulate_engine` runs a complete multivariate simulation over a deterministic dependency-level scheduler (serial == parallel at any thread count), handling conditioning, nonstationary lag transforms, zoned multi-TI domains, and the fallback/boundary logic. `cleanup_pass_engine` runs one deterministic post-processing (Gauss-Seidel) pass over a position-oriented dependency DAG. `simulate_node` is the shared per-node worker. |
| `src/short_vec.rs` | Private. `ShortVec<const N: usize>([f64; N])` — a stack-allocated, SIMD-aligned fixed-size vector. Used by `field.rs` to specialise 2D vs 3D at compile time. |

**Python surface:** the 9 original field / kriging / variogram functions
(`summate`, `summate_incompr`, `summate_fourier`, `calc_field_krige`,
`calc_field_krige_and_variance`, `variogram_structured`,
`variogram_ma_structured`, `variogram_directional`, `variogram_unstructured`)
plus the MPS exports: the `mps_dist_block_*` kernels (categorical / l1 / l2 /
lp / variation, with `_masked` and categorical `_rayon` variants),
`mps_scan_node` and `mps_scan_node_cat`, the full-simulation `mps_simulate`,
the post-processing `mps_cleanup_pass`, and `mps_api_version`. The exact
required set is pinned in `gstools/mps/_core.py::_REQUIRED_EXPORTS`.

**Data flow across the boundary:**
```
Python numpy array
  → PyReadonlyArray{1,2}<f64>   (zero-copy, read-only borrow)
  → .as_array()                 (strip PyO3 wrapper → ndarray ArrayView)
  → pure Rust computation       (ndarray + rayon, no Python objects)
  → ndarray::Array.into_pyarray(py)  (hand ownership back to Python)
  → Bound<'py, PyArray{1,2}<f64>>
```

## Coding conventions

These match the existing codebase — keep them when adding new code:

- **Edition 2021**, `#![warn(missing_docs)]` on the crate root.
- **Public functions must have doc comments** (`///`) with `# Arguments` and
  `# Returns` sections matching the existing style in `field.rs`.
- **No `unwrap()` on recoverable errors** in public functions. `unwrap()` is
  acceptable only on infallible setup (e.g. `ThreadPoolBuilder::new().build().unwrap()`).
- **`?` for error propagation** in functions returning `PyResult`.
- **Accept `ArrayView`, not owned `Array`** in pure-Rust functions. Ownership
  conversion happens only in the `lib.rs` wrappers.
- **`Option<usize>` for `num_threads`** — always pass through to
  `ThreadPoolBuilder` via `.unwrap_or(rayon::current_num_threads())`.
- **Const generics for dimensionality** — new field functions should follow the
  `match pos.dim().0 { 2 => inner::<2>(...), 3 => inner::<3>(...), _ => panic!(...) }`
  pattern from `field.rs:177-181`.
- **Traits for algorithm variants** — new estimator or distance types belong as
  `impl` blocks on a new struct, behind the `Estimator` or `Distance` trait in
  `variogram.rs`, selected via the existing macros.
- **`#[allow(clippy::too_many_arguments)]`** is acceptable on PyO3 wrappers
  whose signature is fixed by the mathematical API.
- **`#[repr(align(16))]`** on any new stack-allocated vector type (SIMD).

## Dependency versions (do not change without reason)

```toml
pyo3  = { version = "0.28", features = ["abi3-py38", "extension-module"] }
numpy = "0.28"
ndarray = { version = "0.17", features = ["rayon", "approx"] }
rayon = "1.10"
```

## Edit → test round-trip

1. Edit `.rs` file.
2. `cargo clippy -- -D warnings` (zero warnings required).
3. `env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV maturin develop`.
4. Run a Python smoke-test or the GSTools test suite:
   ```bash
   cd ../GSTools && pytest tests/
   ```
5. For performance-sensitive changes, run `cargo bench` before and after.
