# Rust MPS Backend (Phase 1–3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Python/NumPy inner loops of Direct Sampling with Rust kernels in `gstools_core`, recovering the 52.7 % of field-scale simulation time consumed by `_dist_block` + `vec_categorical_dist` (Phase 1), then `_scan_window` (Phase 2, +9.9 %), then `_select_neighbors` (Phase 3, +5.8 %).

**Architecture:** A new `src/mps.rs` module in GSTools-Core exposes fused gather + distance kernels via PyO3. Python's `_scan_for_match` (in `gstools.mps.scan`) dispatches to Rust when `USE_GSTOOLS_CORE` and `_GSTOOLS_CORE_AVAIL` are both True and `ti_has_nan` is False — the same two-flag pattern already used in `variogram.py`, `krige/base.py`, and `field/generator.py`. The pure-Python path remains fully intact as fallback.

**Tech Stack:** Rust ndarray 0.17, PyO3 0.28, no rayon (parallelism lives at the node level in `runner.py`, not the block level). Python: NumPy, `gstools.config`. Build: `maturin develop --release`. Tests: `cargo test`, `pytest tests/test_mps_rust_backend.py`, `asv run 'HEAD^!'`.

---

## Baseline Benchmark (ASV, Python 3.14, Apple Silicon)

From `.asv/results/ufz543326.intranet.ufz.de/f4062e2f-*.json` — these are the targets to beat:

| Case | Config | Pure-Python time |
|---|---|---|
| `cat_dsbc_small` | TI 40×40, SG 20×20, n=8 | **0.198 s** |
| `cat_dsbc_medium` | TI 60×60, SG 30×30, n=12 | **0.796 s** |
| `cat_dsbc_large` | TI 120×120, SG 40×40, n=16 | **3.594 s** |
| `cat_dsbc_hiscan` | TI 60×60, SG 30×30, n=12, f=0.8 | **1.203 s** |
| `cat_dsbc_highk` | TI 60×60, SG 30×30, n=24 | **1.382 s** |
| `cont_l1_medium` | TI 60×60, SG 30×30, n=12, L1 | **0.770 s** |
| `cat_ds_medium` | TI 60×60, SG 30×30, n=12, DS | **0.816 s** |
| `cat_dsbc_cond` | TI 60×60, SG 30×30, n=12, cond | **0.549 s** |

## Hot-Path Priority (from `mps_benchmark_report/MPS_RUST_PRIORITY.md`)

| Phase | Function | Module | Field-scale % | Expected speedup |
|---|---|---|---|---|
| **1a** | `_dist_block` closure | `scan.py` | 32.5 % | 10–50× |
| **1b** | `vec_categorical_dist` | `distance.py` | 20.2 % | called inside 1a |
| Phase 1 total | | | **52.7 %** | |
| **2** | `_scan_window` | `scan.py` | 9.9 % | 5–20× |
| **3** | `_select_neighbors` | `neighbors.py` | 5.8 % | 5–10× |

`_dist_block` fuses the gather (`all_de_ti` materialization) with the distance call. A single Rust kernel replaces both 1a and 1b together.

---

## Global Constraints

- All pure-Rust functions accept `ArrayView` (not owned arrays); ownership conversion only in `lib.rs` `#[pyfunction]` wrappers — matches existing codebase pattern.
- `#![warn(missing_docs)]` is crate-wide. Every new `pub fn` requires `///` doc comments with `# Arguments` and `# Returns` sections.
- No `unwrap()` on recoverable errors in public functions. Panics are acceptable only for broken invariants (e.g. mismatched array lengths that represent caller bugs).
- Python dispatch guard: `config.USE_GSTOOLS_CORE and config._GSTOOLS_CORE_AVAIL` — both flags required (matches `variogram.py:60`, `krige/base.py:44`).
- Rust path restriction: only when `not cfg.ti_has_nan`. NaN-handling is a future PR.
- Edition 2021. No rayon dependency on new code (parallelism is at node level in `runner.py`).
- `cargo clippy --all-targets --all-features -- -D warnings` must emit zero warnings before each commit.
- Build for benchmarking: `env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV /Users/lealroja/Documents/UFZ/RUST/.venv/bin/maturin develop --release`
- Build for iteration: `env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV maturin develop`

---

## File Structure

**GSTools-Core (`/Users/lealroja/Documents/UFZ/MPS-Tools/GSTools-Core/`):**

| Action | Path | Responsibility |
|---|---|---|
| **Create** | `src/mps.rs` | Phase 1: `dist_block_categorical`, `dist_block_l1`, `dist_block_l2`, `dist_block_lp`; Phase 2: `scan_node_categorical`; Phase 3: `select_neighbors` |
| **Modify** | `src/lib.rs` | Add `pub mod mps;`, import new fns, add PyO3 wrappers: `mps_dist_block_cat`, `mps_dist_block_l1`, `mps_dist_block_l2`, `mps_dist_block_lp` (Phase 1); `mps_scan_node_cat` (Phase 2) |

**GSTools (`/Users/lealroja/Documents/UFZ/MPS-Tools/GSTools/`):**

| Action | Path | Responsibility |
|---|---|---|
| **Modify** | `src/gstools/mps/scan.py` | Add `var_categorical`, `var_p_norm`, `var_d_max`, `ti_flat_f64` fields to `_ScanConfig`; add `gstools_core` import guard; rewrite `_dist_block` with Rust dispatch for categorical + Lp; Phase 2: add Rust path bypassing `_scan_window` for univariate categorical |
| **Modify** | `src/gstools/mps/simulate.py` | Populate the four new `_ScanConfig` fields at engine construction time |
| **Create** | `tests/test_mps_rust_backend.py` | Integration tests comparing Rust vs Python output |
| **Modify** | `benchmarks/benchmark_mps.py` | Add `BACKENDS = ("python", "core")` param to `DirectSamplingBenchmarks` |

---

## Task 1: Scaffold `src/mps.rs` and wire into `src/lib.rs`

**Files:**
- Create: `src/mps.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Create `src/mps.rs` with module doc and imports**

  ```rust
  // src/mps.rs
  //! MPS (Multiple Point Statistics) kernels for Direct Sampling.
  //!
  //! Fused gather + distance functions that replace the Python `_dist_block`
  //! closure in `gstools/mps/scan.py`. Each kernel reads TI candidates via
  //! flat-index arithmetic and computes weighted distances in one pass,
  //! avoiding the temporary `all_de_ti` allocation of the Python path.

  use ndarray::{Array1, ArrayView1};
  ```

- [ ] **Step 2: Add `pub mod mps;` to `src/lib.rs`**

  In `src/lib.rs`, after the existing `mod short_vec;` line:

  ```rust
  pub mod mps;
  ```

- [ ] **Step 3: Compile check**

  ```bash
  cd /Users/lealroja/Documents/UFZ/MPS-Tools/GSTools-Core
  cargo clippy --all-targets --all-features -- -D warnings
  ```

  Expected: 0 warnings, 0 errors.

- [ ] **Step 4: Commit**

  ```bash
  cd /Users/lealroja/Documents/UFZ/MPS-Tools/GSTools-Core
  git add src/mps.rs src/lib.rs
  git commit -m "feat(mps): scaffold src/mps.rs — empty module wired into lib.rs"
  ```

---

## Task 2: `dist_block_categorical` — fused gather + categorical distance

**Files:**
- Modify: `src/mps.rs`

**Interfaces:**
- Produces: `pub fn dist_block_categorical(de_sim: ArrayView1<'_, f64>, ti_flat: ArrayView1<'_, f64>, base_flat: ArrayView1<'_, i64>, lag_flat: ArrayView1<'_, i64>, node_weights: ArrayView1<'_, f64>) -> Array1<f64>`

**Data-flow note:** In Python, `_dist_block` computes `base = y_blk @ ti_strides` (flat TI index for each candidate anchor) and `flat_il = int_lags @ ti_strides` (flat lag offsets). The Rust kernel receives both pre-computed arrays and accesses `ti_flat[base + lag]` for each candidate `b` and each lag `i`.

- [ ] **Step 1: Write the failing Rust unit tests**

  Add `#[cfg(test)] mod tests` to `src/mps.rs`:

  ```rust
  #[cfg(test)]
  mod tests {
      use super::*;
      use approx::assert_abs_diff_eq;
      use ndarray::array;

      #[test]
      fn categorical_exact_match() {
          // TI: [3.0, 7.0], candidate base=0, lag=0 → ti[0]=3.0 == de[0]=3.0 → d=0.0
          let de_sim = array![3.0_f64];
          let ti_flat = array![3.0_f64, 7.0_f64];
          let base_flat = array![0_i64];
          let lag_flat = array![0_i64];
          let weights = array![1.0_f64];
          let d = dist_block_categorical(
              de_sim.view(), ti_flat.view(),
              base_flat.view(), lag_flat.view(), weights.view(),
          );
          assert_eq!(d.len(), 1);
          assert_abs_diff_eq!(d[0], 0.0, epsilon = 1e-12);
      }

      #[test]
      fn categorical_full_mismatch() {
          // base=1: ti[1+0]=7.0 != de[0]=3.0 → d=1.0
          let de_sim = array![3.0_f64];
          let ti_flat = array![3.0_f64, 7.0_f64];
          let base_flat = array![1_i64];
          let lag_flat = array![0_i64];
          let weights = array![1.0_f64];
          let d = dist_block_categorical(
              de_sim.view(), ti_flat.view(),
              base_flat.view(), lag_flat.view(), weights.view(),
          );
          assert_abs_diff_eq!(d[0], 1.0, epsilon = 1e-12);
      }

      #[test]
      fn categorical_partial_match() {
          // TI=[1,2,3], de=[1,9], base=0, lags=[0,1], weights=[0.5,0.5]
          // lag 0: ti[0]=1 == de[0]=1 → 0 mismatch; lag 1: ti[1]=2 != de[1]=9 → 1 mismatch
          // d = 0.0*0.5 + 1.0*0.5 = 0.5
          let de_sim = array![1.0_f64, 9.0_f64];
          let ti_flat = array![1.0_f64, 2.0_f64, 3.0_f64];
          let base_flat = array![0_i64];
          let lag_flat = array![0_i64, 1_i64];
          let weights = array![0.5_f64, 0.5_f64];
          let d = dist_block_categorical(
              de_sim.view(), ti_flat.view(),
              base_flat.view(), lag_flat.view(), weights.view(),
          );
          assert_abs_diff_eq!(d[0], 0.5, epsilon = 1e-12);
      }

      #[test]
      fn categorical_two_candidates() {
          // TI=[5,5,7], de=[5], lag=[0], weights=[1.0]
          // base 0: ti[0]=5 == 5 → d=0.0; base 2: ti[2]=7 != 5 → d=1.0
          let de_sim = array![5.0_f64];
          let ti_flat = array![5.0_f64, 5.0_f64, 7.0_f64];
          let base_flat = array![0_i64, 2_i64];
          let lag_flat = array![0_i64];
          let weights = array![1.0_f64];
          let d = dist_block_categorical(
              de_sim.view(), ti_flat.view(),
              base_flat.view(), lag_flat.view(), weights.view(),
          );
          assert_abs_diff_eq!(d[0], 0.0, epsilon = 1e-12);
          assert_abs_diff_eq!(d[1], 1.0, epsilon = 1e-12);
      }

      #[test]
      fn categorical_negative_lag() {
          // TI=[1,2,3], de=[2], base=2, lag=-1 → ti[2-1]=ti[1]=2 == 2 → d=0.0
          let de_sim = array![2.0_f64];
          let ti_flat = array![1.0_f64, 2.0_f64, 3.0_f64];
          let base_flat = array![2_i64];
          let lag_flat = array![-1_i64];
          let weights = array![1.0_f64];
          let d = dist_block_categorical(
              de_sim.view(), ti_flat.view(),
              base_flat.view(), lag_flat.view(), weights.view(),
          );
          assert_abs_diff_eq!(d[0], 0.0, epsilon = 1e-12);
      }
  }
  ```

- [ ] **Step 2: Run tests — expect compile error**

  ```bash
  cargo test mps::tests 2>&1 | head -10
  ```

  Expected: `error[E0425]: cannot find function dist_block_categorical in this scope`

- [ ] **Step 3: Implement `dist_block_categorical`**

  Add to `src/mps.rs` before the `#[cfg(test)]` block:

  ```rust
  /// Fused gather + weighted categorical distance over a block of TI candidates.
  ///
  /// For each candidate `b`, reads TI values at flat positions
  /// `base_flat[b] + lag_flat[i]` for each lag `i` and computes the weighted
  /// categorical mismatch against `de_sim`:
  ///
  /// `dist[b] = Σᵢ node_weights[i] · (de_sim[i] ≠ ti_flat[base_flat[b] + lag_flat[i]])`
  ///
  /// The caller guarantees all `base_flat[b] + lag_flat[i]` indices are in-bounds.
  /// The Python TI search-window construction (`_intersect_search_windows` in
  /// `neighbors.py`) enforces this invariant before calling this function.
  ///
  /// # Arguments
  ///
  /// * `de_sim` – Data-event values at the current simulation node, shape `(n,)`.
  ///   Integer categories stored as `f64` (same representation as the sim grid).
  /// * `ti_flat` – Flattened C-order TI array, shape `(ti_size,)`.
  ///   Integer categories stored as `f64` (pre-cast by `_DirectSamplingEngine`).
  /// * `base_flat` – Flat base index for each candidate anchor, shape `(B,)`.
  ///   Computed by Python as `(y_blk @ ti_strides).astype(np.int64)`.
  /// * `lag_flat` – Flat lag offsets, shape `(n,)`.
  ///   Computed by Python as `(int_lags @ ti_strides).astype(np.int64)`.
  /// * `node_weights` – Precomputed spatial-decay weights normalized to sum 1, shape `(n,)`.
  ///
  /// # Returns
  ///
  /// Distance array of shape `(B,)` with values in `[0.0, 1.0]`.
  pub fn dist_block_categorical(
      de_sim: ArrayView1<'_, f64>,
      ti_flat: ArrayView1<'_, f64>,
      base_flat: ArrayView1<'_, i64>,
      lag_flat: ArrayView1<'_, i64>,
      node_weights: ArrayView1<'_, f64>,
  ) -> Array1<f64> {
      Array1::from_iter(base_flat.iter().map(|&base| {
          de_sim
              .iter()
              .zip(lag_flat.iter())
              .zip(node_weights.iter())
              .fold(0.0_f64, |d, ((&de_val, &lag), &wt)| {
                  let ti_val = ti_flat[(base + lag) as usize];
                  // Integer categories are stored as exact f64 values (0.0, 1.0, …).
                  // A gap > 0.5 reliably identifies a categorical mismatch while
                  // being robust to any future float representation of small integers.
                  d + if (de_val - ti_val).abs() > 0.5 { wt } else { 0.0 }
              })
      }))
  }
  ```

- [ ] **Step 4: Run tests — expect all pass**

  ```bash
  cargo test mps::tests -- --nocapture
  ```

  Expected: 5 tests pass.

- [ ] **Step 5: Clippy check**

  ```bash
  cargo clippy --all-targets --all-features -- -D warnings
  ```

  Expected: 0 warnings.

- [ ] **Step 6: Commit**

  ```bash
  git add src/mps.rs
  git commit -m "feat(mps): dist_block_categorical — fused gather + categorical distance, 5 unit tests"
  ```

---

## Task 3: `dist_block_l1`, `dist_block_l2`, `dist_block_lp`

**Files:**
- Modify: `src/mps.rs`

**Interfaces:**
- Produces:
  - `pub fn dist_block_l1(..., d_max: f64) -> Array1<f64>`
  - `pub fn dist_block_l2(..., d_max: f64) -> Array1<f64>`
  - `pub fn dist_block_lp(..., d_max: f64, p: f64) -> Array1<f64>`
- Consumes: same `(de_sim, ti_flat, base_flat, lag_flat, node_weights)` array types as Task 2.

- [ ] **Step 1: Write failing unit tests**

  Add to the `mod tests` block in `src/mps.rs`:

  ```rust
  #[test]
  fn l1_single_lag() {
      // |5.0 - 3.0| / 4.0 * 1.0 = 0.5
      let de_sim = array![5.0_f64];
      let ti_flat = array![3.0_f64];
      let base_flat = array![0_i64];
      let lag_flat = array![0_i64];
      let weights = array![1.0_f64];
      let d = dist_block_l1(
          de_sim.view(), ti_flat.view(),
          base_flat.view(), lag_flat.view(), weights.view(), 4.0,
      );
      assert_abs_diff_eq!(d[0], 0.5, epsilon = 1e-12);
  }

  #[test]
  fn l1_two_candidates() {
      // TI=[0,2,4], de=[4,0], weights=[0.5,0.5], d_max=4.0
      // base 0, lags=[0,1]: ti=[0,2]; |4-0|/4*0.5 + |0-2|/4*0.5 = 0.5 + 0.25 = 0.75
      // base 1, lags=[0,1]: ti=[2,4]; |4-2|/4*0.5 + |0-4|/4*0.5 = 0.25 + 0.5 = 0.75
      let de_sim = array![4.0_f64, 0.0_f64];
      let ti_flat = array![0.0_f64, 2.0_f64, 4.0_f64];
      let base_flat = array![0_i64, 1_i64];
      let lag_flat = array![0_i64, 1_i64];
      let weights = array![0.5_f64, 0.5_f64];
      let d = dist_block_l1(
          de_sim.view(), ti_flat.view(),
          base_flat.view(), lag_flat.view(), weights.view(), 4.0,
      );
      assert_abs_diff_eq!(d[0], 0.75, epsilon = 1e-12);
      assert_abs_diff_eq!(d[1], 0.75, epsilon = 1e-12);
  }

  #[test]
  fn l2_single_lag() {
      // √(1.0 * ((5-3)/4)²) = √(0.25) = 0.5
      let de_sim = array![5.0_f64];
      let ti_flat = array![3.0_f64];
      let base_flat = array![0_i64];
      let lag_flat = array![0_i64];
      let weights = array![1.0_f64];
      let d = dist_block_l2(
          de_sim.view(), ti_flat.view(),
          base_flat.view(), lag_flat.view(), weights.view(), 4.0,
      );
      assert_abs_diff_eq!(d[0], 0.5, epsilon = 1e-12);
  }

  #[test]
  fn lp_p1_matches_l1() {
      // lp(p=1) must equal l1 for same inputs
      let de_sim = array![5.0_f64];
      let ti_flat = array![3.0_f64];
      let base_flat = array![0_i64];
      let lag_flat = array![0_i64];
      let weights = array![1.0_f64];
      let d_l1 = dist_block_l1(
          de_sim.view(), ti_flat.view(),
          base_flat.view(), lag_flat.view(), weights.view(), 4.0,
      );
      let d_lp = dist_block_lp(
          de_sim.view(), ti_flat.view(),
          base_flat.view(), lag_flat.view(), weights.view(), 4.0, 1.0,
      );
      assert_abs_diff_eq!(d_l1[0], d_lp[0], epsilon = 1e-12);
  }

  #[test]
  fn lp_p2_matches_l2() {
      let de_sim = array![5.0_f64];
      let ti_flat = array![3.0_f64];
      let base_flat = array![0_i64];
      let lag_flat = array![0_i64];
      let weights = array![1.0_f64];
      let d_l2 = dist_block_l2(
          de_sim.view(), ti_flat.view(),
          base_flat.view(), lag_flat.view(), weights.view(), 4.0,
      );
      let d_lp = dist_block_lp(
          de_sim.view(), ti_flat.view(),
          base_flat.view(), lag_flat.view(), weights.view(), 4.0, 2.0,
      );
      assert_abs_diff_eq!(d_l2[0], d_lp[0], epsilon = 1e-10);
  }
  ```

- [ ] **Step 2: Run failing tests**

  ```bash
  cargo test mps::tests::l1 2>&1 | head -5
  cargo test mps::tests::l2 2>&1 | head -5
  cargo test mps::tests::lp 2>&1 | head -5
  ```

  Expected: compile errors (functions undefined).

- [ ] **Step 3: Implement all three functions**

  Add to `src/mps.rs`, after `dist_block_categorical`:

  ```rust
  /// Fused gather + weighted L1 distance over a block of TI candidates.
  ///
  /// `dist[b] = Σᵢ node_weights[i] · |de_sim[i] − ti_flat[base_flat[b] + lag_flat[i]]| / d_max`
  ///
  /// # Arguments
  ///
  /// * `d_max` – TI value range (max − min over finite cells); from `Variable.d_max`.
  ///   `Variable._data_range` guarantees this is positive.
  ///
  /// # Returns
  ///
  /// Distance array of shape `(B,)`, values nominally in `[0, 1]`.
  pub fn dist_block_l1(
      de_sim: ArrayView1<'_, f64>,
      ti_flat: ArrayView1<'_, f64>,
      base_flat: ArrayView1<'_, i64>,
      lag_flat: ArrayView1<'_, i64>,
      node_weights: ArrayView1<'_, f64>,
      d_max: f64,
  ) -> Array1<f64> {
      Array1::from_iter(base_flat.iter().map(|&base| {
          de_sim
              .iter()
              .zip(lag_flat.iter())
              .zip(node_weights.iter())
              .fold(0.0_f64, |d, ((&de_val, &lag), &wt)| {
                  let ti_val = ti_flat[(base + lag) as usize];
                  d + wt * (de_val - ti_val).abs() / d_max
              })
      }))
  }

  /// Fused gather + weighted L2 distance over a block of TI candidates.
  ///
  /// `dist[b] = √(Σᵢ node_weights[i] · ((de_sim[i] − ti_flat[base_flat[b] + lag_flat[i]]) / d_max)²)`
  ///
  /// # Arguments
  ///
  /// * `d_max` – TI value range for normalization. Must be positive.
  ///
  /// # Returns
  ///
  /// Distance array of shape `(B,)`.
  pub fn dist_block_l2(
      de_sim: ArrayView1<'_, f64>,
      ti_flat: ArrayView1<'_, f64>,
      base_flat: ArrayView1<'_, i64>,
      lag_flat: ArrayView1<'_, i64>,
      node_weights: ArrayView1<'_, f64>,
      d_max: f64,
  ) -> Array1<f64> {
      Array1::from_iter(base_flat.iter().map(|&base| {
          de_sim
              .iter()
              .zip(lag_flat.iter())
              .zip(node_weights.iter())
              .fold(0.0_f64, |acc, ((&de_val, &lag), &wt)| {
                  let diff = (de_val - ti_flat[(base + lag) as usize]) / d_max;
                  acc + wt * diff * diff
              })
              .sqrt()
      }))
  }

  /// Fused gather + weighted Lp distance over a block of TI candidates.
  ///
  /// `dist[b] = (Σᵢ node_weights[i] · |de_sim[i] − ti_flat[base_flat[b] + lag_flat[i]]|ᵖ / d_maxᵖ)^(1/p)`
  ///
  /// For `p = 1` this is equivalent to `dist_block_l1`; for `p = 2` to `dist_block_l2`.
  /// Prefer those specialized functions for `p ∈ {1, 2}` as they avoid `f64::powf`.
  ///
  /// # Arguments
  ///
  /// * `d_max` – TI value range for normalization. Must be positive.
  /// * `p` – Lp norm exponent; from `Variable.p_norm`. Must be positive.
  ///
  /// # Returns
  ///
  /// Distance array of shape `(B,)`.
  pub fn dist_block_lp(
      de_sim: ArrayView1<'_, f64>,
      ti_flat: ArrayView1<'_, f64>,
      base_flat: ArrayView1<'_, i64>,
      lag_flat: ArrayView1<'_, i64>,
      node_weights: ArrayView1<'_, f64>,
      d_max: f64,
      p: f64,
  ) -> Array1<f64> {
      let inv_p = p.recip();
      Array1::from_iter(base_flat.iter().map(|&base| {
          de_sim
              .iter()
              .zip(lag_flat.iter())
              .zip(node_weights.iter())
              .fold(0.0_f64, |acc, ((&de_val, &lag), &wt)| {
                  let diff = (de_val - ti_flat[(base + lag) as usize]).abs() / d_max;
                  acc + wt * diff.powf(p)
              })
              .powf(inv_p)
      }))
  }
  ```

- [ ] **Step 4: Run all tests**

  ```bash
  cargo test mps::tests -- --nocapture
  ```

  Expected: 10 tests pass.

- [ ] **Step 5: Clippy check**

  ```bash
  cargo clippy --all-targets --all-features -- -D warnings
  ```

- [ ] **Step 6: Commit**

  ```bash
  git add src/mps.rs
  git commit -m "feat(mps): dist_block_l1/l2/lp Rust kernels + unit tests"
  ```

---

## Task 4: PyO3 wrappers in `src/lib.rs`

**Files:**
- Modify: `src/lib.rs`

**Interfaces:**
- Produces (Python names): `mps_dist_block_cat`, `mps_dist_block_l1`, `mps_dist_block_l2`, `mps_dist_block_lp`
- Consumes: `crate::mps::{dist_block_categorical, dist_block_l1, dist_block_l2, dist_block_lp}`

- [ ] **Step 1: Write the Python smoke-test (will fail until step 3)**

  Save as `/tmp/test_mps_core_smoke.py`:

  ```python
  """Smoke-test: verify gstools_core exports the four mps_dist_block_* functions."""
  import numpy as np
  import gstools_core

  de_sim    = np.array([1.0, 2.0], dtype=np.float64)
  ti_flat   = np.array([1.0, 2.0, 3.0, 0.0, 1.0, 2.0], dtype=np.float64)
  base_flat = np.array([0, 3], dtype=np.int64)
  lag_flat  = np.array([0, 1], dtype=np.int64)
  weights   = np.array([0.5, 0.5], dtype=np.float64)

  # base=0: ti[0,1]=[1,2] vs de=[1,2] → both match → d=0.0
  # base=3: ti[3,4]=[0,1] vs de=[1,2] → both mismatch → d=1.0
  d = gstools_core.mps_dist_block_cat(de_sim, ti_flat, base_flat, lag_flat, weights)
  assert d.shape == (2,), f"shape: {d.shape}"
  assert abs(d[0] - 0.0) < 1e-12 and abs(d[1] - 1.0) < 1e-12, f"cat: {d}"
  print("mps_dist_block_cat: OK", d)

  d1 = gstools_core.mps_dist_block_l1(de_sim, ti_flat, base_flat, lag_flat, weights, 2.0)
  assert d1.shape == (2,), f"l1 shape: {d1.shape}"
  print("mps_dist_block_l1: OK", d1)

  d2 = gstools_core.mps_dist_block_l2(de_sim, ti_flat, base_flat, lag_flat, weights, 2.0)
  assert d2.shape == (2,), f"l2 shape: {d2.shape}"
  print("mps_dist_block_l2: OK", d2)

  dp = gstools_core.mps_dist_block_lp(de_sim, ti_flat, base_flat, lag_flat, weights, 2.0, 3.0)
  assert dp.shape == (2,), f"lp shape: {dp.shape}"
  print("mps_dist_block_lp: OK", dp)
  print("All smoke tests passed.")
  ```

  Run (should fail):

  ```bash
  /Users/lealroja/miniconda3/envs/gstools/bin/python /tmp/test_mps_core_smoke.py
  ```

  Expected: `AttributeError: module 'gstools_core' has no attribute 'mps_dist_block_cat'`

- [ ] **Step 2: Add `use crate::mps::...` import inside the `mod gstools_core { }` block in `src/lib.rs`**

  Find the `use` block inside `mod gstools_core { ... }` and add:

  ```rust
  use crate::mps::{dist_block_categorical, dist_block_l1, dist_block_l2, dist_block_lp};
  ```

- [ ] **Step 3: Add four `#[pyfunction]` wrappers inside `mod gstools_core { }` in `src/lib.rs`**

  After the last existing `#[pyfunction]` (the `variogram_unstructured` wrapper), add:

  ```rust
  #[pyfunction(name = "mps_dist_block_cat")]
  fn mps_dist_block_cat_py<'py>(
      py: Python<'py>,
      de_sim: PyReadonlyArray1<f64>,
      ti_flat: PyReadonlyArray1<f64>,
      base_flat: PyReadonlyArray1<i64>,
      lag_flat: PyReadonlyArray1<i64>,
      node_weights: PyReadonlyArray1<f64>,
  ) -> Bound<'py, PyArray1<f64>> {
      dist_block_categorical(
          de_sim.as_array(),
          ti_flat.as_array(),
          base_flat.as_array(),
          lag_flat.as_array(),
          node_weights.as_array(),
      )
      .into_pyarray(py)
  }

  #[pyfunction(name = "mps_dist_block_l1")]
  #[allow(clippy::too_many_arguments)]
  fn mps_dist_block_l1_py<'py>(
      py: Python<'py>,
      de_sim: PyReadonlyArray1<f64>,
      ti_flat: PyReadonlyArray1<f64>,
      base_flat: PyReadonlyArray1<i64>,
      lag_flat: PyReadonlyArray1<i64>,
      node_weights: PyReadonlyArray1<f64>,
      d_max: f64,
  ) -> Bound<'py, PyArray1<f64>> {
      dist_block_l1(
          de_sim.as_array(),
          ti_flat.as_array(),
          base_flat.as_array(),
          lag_flat.as_array(),
          node_weights.as_array(),
          d_max,
      )
      .into_pyarray(py)
  }

  #[pyfunction(name = "mps_dist_block_l2")]
  #[allow(clippy::too_many_arguments)]
  fn mps_dist_block_l2_py<'py>(
      py: Python<'py>,
      de_sim: PyReadonlyArray1<f64>,
      ti_flat: PyReadonlyArray1<f64>,
      base_flat: PyReadonlyArray1<i64>,
      lag_flat: PyReadonlyArray1<i64>,
      node_weights: PyReadonlyArray1<f64>,
      d_max: f64,
  ) -> Bound<'py, PyArray1<f64>> {
      dist_block_l2(
          de_sim.as_array(),
          ti_flat.as_array(),
          base_flat.as_array(),
          lag_flat.as_array(),
          node_weights.as_array(),
          d_max,
      )
      .into_pyarray(py)
  }

  #[pyfunction(name = "mps_dist_block_lp")]
  #[allow(clippy::too_many_arguments)]
  fn mps_dist_block_lp_py<'py>(
      py: Python<'py>,
      de_sim: PyReadonlyArray1<f64>,
      ti_flat: PyReadonlyArray1<f64>,
      base_flat: PyReadonlyArray1<i64>,
      lag_flat: PyReadonlyArray1<i64>,
      node_weights: PyReadonlyArray1<f64>,
      d_max: f64,
      p: f64,
  ) -> Bound<'py, PyArray1<f64>> {
      dist_block_lp(
          de_sim.as_array(),
          ti_flat.as_array(),
          base_flat.as_array(),
          lag_flat.as_array(),
          node_weights.as_array(),
          d_max,
          p,
      )
      .into_pyarray(py)
  }
  ```

- [ ] **Step 4: Clippy check**

  ```bash
  cargo clippy --all-targets --all-features -- -D warnings
  ```

- [ ] **Step 5: Build release and run smoke-test**

  ```bash
  env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV /Users/lealroja/Documents/UFZ/RUST/.venv/bin/maturin develop --release
  /Users/lealroja/miniconda3/envs/gstools/bin/python /tmp/test_mps_core_smoke.py
  ```

  Expected: `All smoke tests passed.`

- [ ] **Step 6: Commit**

  ```bash
  git add src/lib.rs
  git commit -m "feat(mps): expose dist_block kernels to Python via PyO3 — 4 new gstools_core functions"
  ```

---

## Task 5: Python integration — `_dist_block` dispatch in `scan.py` + `simulate.py`

**Files:**
- Modify: `src/gstools/mps/scan.py`
- Modify: `src/gstools/mps/simulate.py`

**Interfaces:**
- Consumes: `gstools_core.mps_dist_block_cat`, `mps_dist_block_l1`, `mps_dist_block_l2`, `mps_dist_block_lp`
- Produces: `DirectSampling` output bit-identical to Python path for the same seeds

**Key type invariants before calling Rust:**
- `de_sim` (`de_v[v]`): always `float64` (simulation grid is `np.full(..., np.nan)` = float64).
- `ti_flat_f64`: pre-cast float64 copy of `ti_flat[v]` (which is int32 for categorical TIs).
- `base_flat` (`y_blk @ ti_strides[v]`): `np.intp` = `int64` on 64-bit; `.astype(np.int64)` is a no-op.
- `lag_flat` (`flat_il[v]`): `float64` with integer values (from `np.rint`); must cast to `int64`.
- `node_weights` (`precomp_w[v]`): `float64`.

- [ ] **Step 1: Add `var_categorical`, `var_p_norm`, `var_d_max`, `ti_flat_f64` to `_ScanConfig`**

  In `src/gstools/mps/scan.py`, extend the `_ScanConfig` dataclass:

  ```python
  @dataclass(frozen=True)
  class _ScanConfig:
      variables: tuple
      weights: dict
      ti_vars: dict
      ti_flat: dict
      ti_strides: dict
      ti_shape: object
      scan_fraction: float
      threshold: float
      cond_weight: float
      distance_power: float
      vec_distance_var: object
      ti_has_nan: bool
      # New fields for Rust dispatch (populated by _DirectSamplingEngine)
      var_categorical: dict   # {v: bool}
      var_p_norm: dict        # {v: float or None}  — None for categorical / variation
      var_d_max: dict         # {v: float or None}  — None for categorical
      ti_flat_f64: dict       # {v: ndarray float64} — pre-cast flat TI for Rust
  ```

- [ ] **Step 2: Add the `gstools_core` import guard at the top of `scan.py`**

  After the existing imports:

  ```python
  from gstools import config as _mps_config

  if _mps_config._GSTOOLS_CORE_AVAIL:  # pragma: no cover
      from gstools_core import mps_dist_block_cat as _mps_dist_block_cat_gsc
      from gstools_core import mps_dist_block_l1 as _mps_dist_block_l1_gsc
      from gstools_core import mps_dist_block_l2 as _mps_dist_block_l2_gsc
      from gstools_core import mps_dist_block_lp as _mps_dist_block_lp_gsc
  ```

- [ ] **Step 3: Rewrite `_dist_block` inside `_scan_for_match` to dispatch to Rust**

  Inside `_scan_for_match`, immediately after computing `flat_il` and `precomp_w`, add:

  ```python
  _use_rust = (
      _mps_config.USE_GSTOOLS_CORE
      and _mps_config._GSTOOLS_CORE_AVAIL
      and not cfg.ti_has_nan
  )
  ```

  Then replace the `def _dist_block(y_blk):` closure with:

  ```python
  def _dist_block(y_blk):
      d = np.zeros(len(y_blk))
      for v in active_vars:
          base = (y_blk @ cfg.ti_strides[v]).astype(np.int64)
          if _use_rust and (cfg.var_categorical[v] or cfg.var_p_norm[v] is not None):
              lag_i64 = flat_il[v].astype(np.int64)
              if cfg.var_categorical[v]:
                  dist_v = _mps_dist_block_cat_gsc(
                      de_v[v], cfg.ti_flat_f64[v], base, lag_i64, precomp_w[v],
                  )
              elif cfg.var_p_norm[v] == 1.0:
                  dist_v = _mps_dist_block_l1_gsc(
                      de_v[v], cfg.ti_flat_f64[v], base, lag_i64, precomp_w[v],
                      cfg.var_d_max[v],
                  )
              elif cfg.var_p_norm[v] == 2.0:
                  dist_v = _mps_dist_block_l2_gsc(
                      de_v[v], cfg.ti_flat_f64[v], base, lag_i64, precomp_w[v],
                      cfg.var_d_max[v],
                  )
              else:
                  dist_v = _mps_dist_block_lp_gsc(
                      de_v[v], cfg.ti_flat_f64[v], base, lag_i64, precomp_w[v],
                      cfg.var_d_max[v], cfg.var_p_norm[v],
                  )
          else:
              # Python path: handles variation distance and NaN TIs
              all_de_ti = cfg.ti_flat[v][base[:, None] + flat_il[v][None, :]]
              dist_v = cfg.vec_distance_var(
                  v,
                  de_v[v],
                  all_de_ti,
                  cm_v[v],
                  cfg.cond_weight,
                  ln_v[v],
                  weights=precomp_w[v],
                  has_nan=cfg.ti_has_nan,
              )
          d += cfg.weights[v] * dist_v
      if 0.0 < active_w_total < 1.0:
          d /= active_w_total
      if cfg.ti_has_nan:
          center_ok = np.ones(len(y_blk), dtype=bool)
          ys = tuple(y_blk.T)
          for v in scan_targets:
              cv = cfg.ti_vars[v][ys]
              if np.issubdtype(cv.dtype, np.floating):
                  center_ok &= np.isfinite(cv)
          d = np.where(center_ok, d, np.inf)
      return d
  ```

  > **Variation distance** (`var_p_norm[v] is None` and `var_categorical[v] is False`) falls through to the Python path automatically because neither condition in the `if _use_rust and (...)` check is satisfied for it.

- [ ] **Step 4: Populate the four new fields in `_DirectSamplingEngine.__init__` in `simulate.py`**

  In `simulate.py`, immediately before the `self._scan_config = _ScanConfig(...)` call (around line 253), add:

  ```python
  ti_flat_f64 = {
      v.name: np.asarray(self.ti_flat[v.name], dtype=np.float64)
      for v in training_image.variables
  }
  var_categorical = {v.name: v.categorical for v in training_image.variables}
  var_p_norm = {v.name: v.p_norm for v in training_image.variables}
  var_d_max = {v.name: v.d_max for v in training_image.variables}
  ```

  Then update the `_ScanConfig(...)` call to pass the new fields at the end:

  ```python
  self._scan_config = _ScanConfig(
      variables=self.variables,
      weights=self.weights,
      ti_vars=self.ti_vars,
      ti_flat=self.ti_flat,
      ti_strides=self.ti_strides,
      ti_shape=self.ti_shape,
      scan_fraction=self.scan_fraction,
      threshold=self.threshold,
      cond_weight=self.cond_weight,
      distance_power=self.training_image.distance_power,
      vec_distance_var=self.training_image.vec_distance_var,
      ti_has_nan=self.ti_has_nan,
      var_categorical=var_categorical,
      var_p_norm=var_p_norm,
      var_d_max=var_d_max,
      ti_flat_f64=ti_flat_f64,
  )
  ```

- [ ] **Step 5: Verify Python-only path still works (smoke test with Rust disabled)**

  ```bash
  cd /Users/lealroja/Documents/UFZ/MPS-Tools/GSTools
  PYTHONPATH=src /Users/lealroja/miniconda3/envs/gstools/bin/python3 -c "
  import numpy as np, gstools as gs
  gs.config.USE_GSTOOLS_CORE = False
  gx, gy = np.meshgrid(np.arange(60), np.arange(60), indexing='ij')
  ti_data = ((np.sin(gx/5.) + np.sin((gx+gy)/8.)) > 0).astype(np.int32)
  ti = gs.TrainingImage(ti_data, categorical=True)
  model = gs.MPSModel(ti, scan_fraction=0.1, threshold=0.0)
  ds = gs.DirectSampling(model, seed=42)
  field = ds([np.arange(10, dtype=float), np.arange(10, dtype=float)], store=False)
  print('Python path OK, shape:', field.shape, 'unique values:', np.unique(field))
  "
  ```

  Expected: runs without error, shape `(10, 10)`, unique values `[0. 1.]`.

- [ ] **Step 6: Verify Rust path produces bit-identical output for categorical DSBC**

  ```bash
  PYTHONPATH=src /Users/lealroja/miniconda3/envs/gstools/bin/python3 -c "
  import numpy as np, gstools as gs
  from gstools.mps import DirectSampling, MPSModel, TrainingImage
  gx, gy = np.meshgrid(np.arange(60), np.arange(60), indexing='ij')
  ti_data = ((np.sin(gx/5.) + np.sin((gx+gy)/8.)) > 0).astype(np.int32)
  ti = TrainingImage(ti_data, categorical=True)
  model = MPSModel(ti, scan_fraction=0.3, threshold=0.0)
  grid = [np.arange(20, dtype=float), np.arange(20, dtype=float)]

  gs.config.USE_GSTOOLS_CORE = False
  field_py = DirectSampling(model, seed=42)(grid, store=False)

  gs.config.USE_GSTOOLS_CORE = True
  field_rs = DirectSampling(model, seed=42)(grid, store=False)

  ok = np.array_equal(field_py, field_rs)
  print('Bit-identical:', ok)
  if not ok:
      print('Mismatches:', np.sum(field_py != field_rs), '/', field_py.size)
  "
  ```

  Expected: `Bit-identical: True`

- [ ] **Step 7: Run the full MPS test suite**

  ```bash
  cd /Users/lealroja/Documents/UFZ/MPS-Tools/GSTools
  PYTHONPATH=src /Users/lealroja/miniconda3/envs/gstools/bin/python -m pytest tests/test_mps.py -v 2>&1 | tail -20
  ```

  Expected: all existing tests pass (no regressions).

- [ ] **Step 8: Commit (GSTools repo)**

  ```bash
  cd /Users/lealroja/Documents/UFZ/MPS-Tools/GSTools
  git add src/gstools/mps/scan.py src/gstools/mps/simulate.py
  git commit -m "feat(mps): dispatch _dist_block to Rust kernel when USE_GSTOOLS_CORE=True — Phase 1"
  ```

---

## Task 6: Python regression tests

**Files:**
- Create: `tests/test_mps_rust_backend.py` (in GSTools repo)

**Interfaces:**
- Consumes: `gstools.DirectSampling`, `gstools.MPSModel`, `gstools.TrainingImage`, `gstools.config`
- Produces: pytest-compatible test file skipped automatically when `gstools_core` is absent

- [ ] **Step 1: Create `tests/test_mps_rust_backend.py`**

  ```python
  """Regression tests: Rust MPS kernels produce output identical to the Python path.

  Skipped automatically when gstools_core is not installed (e.g. CI without Rust build).
  The flag gs.config.USE_GSTOOLS_CORE is restored after each test by the fixture.
  """

  from __future__ import annotations

  import numpy as np
  import pytest

  try:
      import gstools_core  # noqa: F401
      _CORE_AVAIL = True
  except ImportError:
      _CORE_AVAIL = False

  pytestmark = pytest.mark.skipif(
      not _CORE_AVAIL, reason="gstools_core not installed"
  )

  import gstools as gs
  from gstools.mps import DirectSampling, MPSModel, TrainingImage


  @pytest.fixture(autouse=True)
  def _restore_core_flag():
      """Restore USE_GSTOOLS_CORE after every test that changes it."""
      original = gs.config.USE_GSTOOLS_CORE
      yield
      gs.config.USE_GSTOOLS_CORE = original


  def _cat_ti(shape=(60, 60)):
      gx, gy = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
      return ((np.sin(gx / 5.0) + np.sin((gx + gy) / 8.0)) > 0).astype(np.int32)


  def _cont_ti(shape=(60, 60)):
      gx, gy = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
      return np.sin(gx / 6.0) * np.cos(gy / 8.0)


  def _run(ti_data, categorical, distance, sg_shape, scan_fraction, threshold, seed, use_rust):
      gs.config.USE_GSTOOLS_CORE = use_rust
      ti = TrainingImage(ti_data, categorical=categorical, distance=distance)
      model = MPSModel(ti, scan_fraction=scan_fraction, threshold=threshold)
      ds = DirectSampling(model, seed=seed)
      return ds([np.arange(s, dtype=float) for s in sg_shape], store=False)


  @pytest.mark.parametrize("seed", [42, 1234, 99999])
  def test_categorical_dsbc_medium(seed):
      ti = _cat_ti()
      py = _run(ti, True, "l1", (30, 30), 0.3, 0.0, seed, False)
      rs = _run(ti, True, "l1", (30, 30), 0.3, 0.0, seed, True)
      np.testing.assert_array_equal(py, rs)


  @pytest.mark.parametrize("seed", [42, 1234])
  def test_categorical_ds_mode(seed):
      ti = _cat_ti()
      py = _run(ti, True, "l1", (30, 30), 0.3, 0.1, seed, False)
      rs = _run(ti, True, "l1", (30, 30), 0.3, 0.1, seed, True)
      np.testing.assert_array_equal(py, rs)


  @pytest.mark.parametrize("seed", [42, 1234])
  def test_categorical_hiscan(seed):
      ti = _cat_ti()
      py = _run(ti, True, "l1", (30, 30), 0.8, 0.0, seed, False)
      rs = _run(ti, True, "l1", (30, 30), 0.8, 0.0, seed, True)
      np.testing.assert_array_equal(py, rs)


  @pytest.mark.parametrize("seed", [42, 1234])
  def test_continuous_l1(seed):
      ti = _cont_ti()
      py = _run(ti, False, "l1", (30, 30), 0.3, 0.0, seed, False)
      rs = _run(ti, False, "l1", (30, 30), 0.3, 0.0, seed, True)
      np.testing.assert_allclose(py, rs, rtol=0, atol=1e-12)


  @pytest.mark.parametrize("seed", [42, 1234])
  def test_continuous_l2(seed):
      ti = _cont_ti()
      py = _run(ti, False, "l2", (30, 30), 0.3, 0.0, seed, False)
      rs = _run(ti, False, "l2", (30, 30), 0.3, 0.0, seed, True)
      np.testing.assert_allclose(py, rs, rtol=0, atol=1e-12)


  @pytest.mark.parametrize("seed", [42])
  def test_continuous_l3(seed):
      """Lp kernel with p=3 matches Python vec_lp_dist."""
      ti = _cont_ti()
      py = _run(ti, False, "l3", (20, 20), 0.3, 0.0, seed, False)
      rs = _run(ti, False, "l3", (20, 20), 0.3, 0.0, seed, True)
      np.testing.assert_allclose(py, rs, rtol=0, atol=1e-12)


  def test_variation_distance_falls_back_to_python():
      """Variation distance has no Rust kernel yet; both paths must agree."""
      ti = _cont_ti()
      py = _run(ti, False, "variation", (20, 20), 0.3, 0.0, 42, False)
      rs = _run(ti, False, "variation", (20, 20), 0.3, 0.0, 42, True)
      np.testing.assert_allclose(py, rs, rtol=0, atol=1e-12)


  def test_conditioned_categorical():
      """Conditioning data is preserved bit-identically regardless of backend."""
      ti_data = _cat_ti()
      ti = TrainingImage(ti_data, categorical=True)
      model = MPSModel(ti, scan_fraction=0.3, threshold=0.0)
      sg_shape = (20, 20)
      rng = np.random.RandomState(7)
      xi = rng.randint(0, sg_shape[0], 15).astype(float)
      yi = rng.randint(0, sg_shape[1], 15).astype(float)
      vals = ti_data[xi.astype(int) % 60, yi.astype(int) % 60].astype(float)

      results = {}
      for use_rust in (False, True):
          gs.config.USE_GSTOOLS_CORE = use_rust
          ds = DirectSampling(model, seed=42)
          ds.set_condition([xi, yi], vals)
          results[use_rust] = ds(
              [np.arange(s, dtype=float) for s in sg_shape], store=False
          )
      np.testing.assert_array_equal(results[False], results[True])
  ```

- [ ] **Step 2: Run the tests**

  ```bash
  cd /Users/lealroja/Documents/UFZ/MPS-Tools/GSTools
  PYTHONPATH=src /Users/lealroja/miniconda3/envs/gstools/bin/python -m pytest tests/test_mps_rust_backend.py -v
  ```

  Expected: all 14 tests pass.

- [ ] **Step 3: Commit**

  ```bash
  git add tests/test_mps_rust_backend.py
  git commit -m "test(mps): Rust-backend regression tests — 14 cases covering cat/L1/L2/Lp/variation/cond"
  ```

---

## Task 7: Benchmark BACKENDS extension

**Files:**
- Modify: `benchmarks/benchmark_mps.py` (GSTools repo)

**Interfaces:**
- Consumes: `gstools.config.USE_GSTOOLS_CORE`, ASV `params` / `param_names` / `setup` / `time_*`

- [ ] **Step 1: Locate the BACKENDS pattern in the other benchmark file for reference**

  ```bash
  grep -n "BACKENDS\|backend\|num_threads" \
    /Users/lealroja/Documents/UFZ/MPS-Tools/GSTools/benchmarks/benchmark_two_point_statistics.py \
    | head -20
  ```

- [ ] **Step 2: Add `BACKENDS` constant and update `DirectSamplingBenchmarks`**

  At the top of `DirectSamplingBenchmarks`, change:

  ```python
  # Before:
  params = [DS_CASES]
  param_names = ["case"]
  ```

  to:

  ```python
  BACKENDS = ("python", "core")

  params = [DS_CASES, BACKENDS]
  param_names = ["case", "backend"]
  ```

- [ ] **Step 3: Update `setup` to set `USE_GSTOOLS_CORE` and skip when core is absent**

  ```python
  def setup(self, data, case, backend):
      _check_mps_available()
      import gstools.config as _cfg
      if backend == "core":
          try:
              import gstools_core  # noqa: F401
          except ImportError:
              raise NotImplementedError(
                  "gstools_core not installed; 'core' backend skipped."
              )
          _cfg.USE_GSTOOLS_CORE = True
      else:
          _cfg.USE_GSTOOLS_CORE = False

      spec = _DS_SPECS[case]
      if spec["categorical"]:
          ti_data = data[case]["ti_data"]
      else:
          ti_data = data[case]["ti_data"]
      ti = TrainingImage(
          ti_data,
          categorical=spec["categorical"],
          distance=spec["distance"],
          n_neighbors=spec["n_neighbors"],
      )
      model = MPSModel(ti, scan_fraction=spec["scan_fraction"], threshold=spec["threshold"])
      self.ds = DirectSampling(model)
      self.grid = [np.arange(s, dtype=float) for s in spec["sg_shape"]]
      self.seed = _DS_SEEDS[case]
      if spec["conditioned"]:
          self.ds.set_condition(data[case]["cond_pos"], data[case]["cond_val"])
  ```

- [ ] **Step 4: Update `time_simulate` and `peakmem_simulate` signatures**

  ```python
  def time_simulate(self, data, case, backend):
      self.ds(self.grid, seed=self.seed, store=False)

  def peakmem_simulate(self, data, case, backend):
      self.ds(self.grid, seed=self.seed, store=False)
  ```

- [ ] **Step 5: Run quick ASV benchmark on HEAD**

  ```bash
  cd /Users/lealroja/Documents/UFZ/MPS-Tools/GSTools
  asv run 'HEAD^!' --bench DirectSamplingBenchmarks --quick --show-stderr 2>&1 | tail -40
  ```

  Expected: timing rows for both `python` and `core` backends. `core` should be noticeably faster for `cat_dsbc_*` cases.

- [ ] **Step 6: Commit**

  ```bash
  git add benchmarks/benchmark_mps.py
  git commit -m "bench(mps): add BACKENDS=(python,core) param to DirectSamplingBenchmarks"
  ```

---

## Task 10: Speedup Report

**Files:**
- Create: `benchmarks/tools/generate_speedup_report.py` (GSTools repo)
- Produces: `/Users/lealroja/Documents/UFZ/MPS-Tools/mps_benchmark_report/MPS_RUST_SPEEDUP.md`

**When to run:** After Task 7 is committed and both backends produce results in ASV.

**Goal:** Parse the ASV result JSON and emit a markdown table showing Python vs Rust timings and speedup ratios for every DS benchmark case, so progress is visible at a glance.

- [ ] **Step 1: Run a full (non-quick) ASV benchmark to get stable timings**

  ```bash
  cd /Users/lealroja/Documents/UFZ/MPS-Tools/GSTools
  asv run 'HEAD^!' --bench DirectSamplingBenchmarks --show-stderr 2>&1 | tail -50
  ```

  This takes 5–15 minutes. Progress prints case-by-case. Both `python` and `core` backends run for all 8 cases (16 total runs).

- [ ] **Step 2: Create `benchmarks/tools/generate_speedup_report.py`**

  ```python
  #!/usr/bin/env python3
  """Parse the latest ASV result JSON and write MPS_RUST_SPEEDUP.md.

  Run after 'asv run HEAD^!' with BACKENDS = ('python', 'core') in
  DirectSamplingBenchmarks (Task 7 of the Rust MPS backend plan).

  Usage:
      python benchmarks/tools/generate_speedup_report.py
  """
  import glob
  import json
  import os
  from datetime import datetime

  _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  _ASV_RESULTS = os.path.join(_REPO_ROOT, ".asv", "results")
  _REPORT_PATH = os.path.join(
      os.path.dirname(_REPO_ROOT),
      "mps_benchmark_report",
      "MPS_RUST_SPEEDUP.md",
  )

  _DS_CASES = (
      "cat_dsbc_small",
      "cat_dsbc_medium",
      "cat_dsbc_large",
      "cat_ds_medium",
      "cont_l1_medium",
      "cat_dsbc_hiscan",
      "cat_dsbc_highk",
      "cat_dsbc_cond",
  )

  _CONFIGS = {
      "cat_dsbc_small":  "TI 40×40,   SG 20×20,  n=8,  f=0.3, categorical DSBC",
      "cat_dsbc_medium": "TI 60×60,   SG 30×30,  n=12, f=0.3, categorical DSBC",
      "cat_dsbc_large":  "TI 120×120, SG 40×40,  n=16, f=0.3, categorical DSBC",
      "cat_ds_medium":   "TI 60×60,   SG 30×30,  n=12, f=0.3, categorical DS (t=0.1)",
      "cont_l1_medium":  "TI 60×60,   SG 30×30,  n=12, f=0.3, continuous L1",
      "cat_dsbc_hiscan": "TI 60×60,   SG 30×30,  n=12, f=0.8, categorical DSBC high-scan",
      "cat_dsbc_highk":  "TI 60×60,   SG 30×30,  n=24, f=0.3, categorical DSBC high-k",
      "cat_dsbc_cond":   "TI 60×60,   SG 30×30,  n=12, f=0.3, categorical DSBC conditioned",
  }

  # Pure-Python baseline from f4062e2f (2026-07-29, pre-Rust)
  _BASELINE_PY = {
      "cat_dsbc_small":  0.19788,
      "cat_dsbc_medium": 0.79623,
      "cat_dsbc_large":  3.59360,
      "cat_ds_medium":   0.81555,
      "cont_l1_medium":  0.77040,
      "cat_dsbc_hiscan": 1.20300,
      "cat_dsbc_highk":  1.38200,
      "cat_dsbc_cond":   0.54900,
  }


  def _find_latest_result():
      jsons = [
          f
          for d in glob.glob(os.path.join(_ASV_RESULTS, "*"))
          for f in glob.glob(os.path.join(d, "*.json"))
          if not os.path.basename(f).startswith(("machine", "benchmarks"))
      ]
      if not jsons:
          raise FileNotFoundError(f"No ASV result JSONs in {_ASV_RESULTS}")
      return max(jsons, key=os.path.getmtime)


  def _parse_times(entry):
      """Return {(case, backend): time_or_None} from an ASV result entry."""
      times_flat = entry[0]  # flat list, one per param combo; None = skipped
      params = entry[1]      # [[case_values], [backend_values]] or [[case_values]]

      if len(params) == 1:
          # Old single-param format (no BACKENDS yet) — return python times only
          cases = [s.strip("'") for s in params[0]]
          return {(c, "python"): t for c, t in zip(cases, times_flat)}

      cases = [s.strip("'") for s in params[0]]
      backends = [s.strip("'") for s in params[1]]
      n_backends = len(backends)
      out = {}
      for i, case in enumerate(cases):
          for j, backend in enumerate(backends):
              t = times_flat[i * n_backends + j]
              out[(case, backend)] = t
      return out


  def _fmt_t(t):
      if t is None:
          return "—"
      return f"{t * 1000:.0f} ms" if t < 1.0 else f"{t:.3f} s"


  def _fmt_speedup(py_t, rs_t):
      if py_t is None or rs_t is None or rs_t == 0:
          return "—"
      return f"{py_t / rs_t:.1f}×"


  def _generate(times, result_file, commit):
      lines = [
          "# MPS Rust Backend — Benchmark Speedup Report",
          "",
          f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
          f"ASV result: `{os.path.basename(result_file)}`",
          f"Commit: `{commit}`",
          "",
          "## Simulation Timings: Python vs Rust",
          "",
          "| Case | Config | Python (new run) | Python (baseline) | Rust `core` | Speedup |",
          "|---|---|---|---|---|---|",
      ]
      valid = []
      for case in _DS_CASES:
          py_t = times.get((case, "python"))
          rs_t = times.get((case, "core"))
          base = _BASELINE_PY.get(case)
          lines.append(
              f"| `{case}` | {_CONFIGS[case]} "
              f"| {_fmt_t(py_t)} | {_fmt_t(base)} "
              f"| {_fmt_t(rs_t)} | {_fmt_speedup(py_t, rs_t)} |"
          )
          if py_t and rs_t:
              valid.append((case, py_t, rs_t))

      lines += [""]
      if valid:
          avg = sum(p / r for _, p, r in valid) / len(valid)
          peak_case, peak_py, peak_rs = max(valid, key=lambda x: x[1] / x[2])
          lines += [
              f"**Average speedup (measured cases):** {avg:.1f}×  ",
              f"**Peak speedup:** {peak_py / peak_rs:.1f}× (`{peak_case}`)",
              "",
          ]

      lines += [
          "## Hot-Path Coverage by Phase",
          "",
          "| Phase | Function(s) replaced | Field-scale % | Rust kernels |",
          "|---|---|---|---|",
          "| Phase 1 | `_dist_block` + `vec_categorical_dist` | **52.7 %** | `mps_dist_block_cat`, `mps_dist_block_l1`, `mps_dist_block_l2`, `mps_dist_block_lp` |",
          "| Phase 2 | `_scan_window` | **9.9 %** | `mps_scan_node_cat` |",
          "| Phase 3 | `_select_neighbors` | **5.8 %** | deferred |",
          "| Variation dist. | `vec_variation_dist` | — | Python fallback (no Rust kernel) |",
          "",
          "## Baseline Reference (pure Python, pre-Rust, 2026-07-29)",
          "",
          "From `.asv/results/ufz543326.intranet.ufz.de/f4062e2f-*.json`:",
          "",
          "| Case | Time |",
          "|---|---|",
      ]
      for case, t in _BASELINE_PY.items():
          lines.append(f"| `{case}` | {_fmt_t(t)} |")

      return "\n".join(lines) + "\n"


  def main():
      result_file = _find_latest_result()
      print(f"Reading: {result_file}")

      with open(result_file) as f:
          data = json.load(f)

      key = "benchmark_mps.DirectSamplingBenchmarks.time_simulate"
      if key not in data.get("results", {}):
          raise KeyError(
              f"{key!r} not found.\n"
              "Run: asv run 'HEAD^!' --bench DirectSamplingBenchmarks"
          )

      times = _parse_times(data["results"][key])
      commit = data.get("commit_hash", "unknown")[:12]

      report = _generate(times, result_file, commit)
      os.makedirs(os.path.dirname(_REPORT_PATH), exist_ok=True)
      with open(_REPORT_PATH, "w") as f:
          f.write(report)

      print(f"Report written to: {_REPORT_PATH}")


  if __name__ == "__main__":
      main()
  ```

- [ ] **Step 3: Run the script to produce the report**

  ```bash
  cd /Users/lealroja/Documents/UFZ/MPS-Tools/GSTools
  /Users/lealroja/miniconda3/envs/gstools/bin/python benchmarks/tools/generate_speedup_report.py
  ```

  Expected output:
  ```
  Reading: .asv/results/ufz543326.intranet.ufz.de/<hash>-conda-...json
  Report written to: .../mps_benchmark_report/MPS_RUST_SPEEDUP.md
  ```

  Open the report and verify the speedup column is non-empty for categorical cases. Continuous L1 should also show a speedup. Variation (`cat_ds_medium` uses DS not variation, so it should show a speedup too — double check by looking at its `distance` param in `_DS_SPECS`).

- [ ] **Step 4: Commit the script**

  ```bash
  git add benchmarks/tools/generate_speedup_report.py
  git commit -m "bench(mps): add generate_speedup_report.py for Python-vs-Rust comparison"
  ```

  The report itself (`MPS_RUST_SPEEDUP.md`) lives in `mps_benchmark_report/` outside the GSTools repo — commit it there separately if desired, or regenerate on demand.

---

## Task 8 (Phase 2): Full node scan in Rust — `scan_node_categorical`

*Implement after Task 7 benchmarks confirm Phase 1 speedup. Phase 2 eliminates the remaining Python block loop (9.9 % field-scale) for the univariate categorical case.*

**Files:**
- Modify: `src/mps.rs`
- Modify: `src/lib.rs`
- Modify: `src/gstools/mps/scan.py`

**Goal:** Add `scan_node_categorical` that runs the full DSBC/DS block loop in Rust for univariate categorical. This combines `_scan_window` + `_dist_block` in a single Rust function, eliminating the Python block-iteration overhead.

**Interfaces:**
- Produces: `pub fn scan_node_categorical(lo, win_shape, start, max_scan, threshold, de_sim, ti_flat, ti_strides, lag_flat, node_weights) -> Option<Array1<i64>>`
- Python exports as: `mps_scan_node_cat`

- [ ] **Step 1: Write Rust unit tests for `scan_node_categorical`**

  Add to `mod tests` in `src/mps.rs`:

  ```rust
  #[test]
  fn scan_node_cat_finds_exact_match() {
      // TI (1D): [1,2,3,2,1], window lo=[0], shape=[5], de=[2], lag=[0], w=[1.0]
      // DSBC: first exact match at scan position 1 (ti[1]=2) → return [1]
      let lo = array![0_i64];
      let win_shape = array![5_i64];
      let ti_flat = array![1.0_f64, 2.0_f64, 3.0_f64, 2.0_f64, 1.0_f64];
      let ti_strides = array![1_i64];
      let de_sim = array![2.0_f64];
      let lag_flat = array![0_i64];
      let weights = array![1.0_f64];
      let result = scan_node_categorical(
          lo.view(), win_shape.view(), 0, 5, 0.0,
          de_sim.view(), ti_flat.view(), ti_strides.view(),
          lag_flat.view(), weights.view(),
      );
      assert!(result.is_some());
      assert_eq!(result.unwrap()[0], 1);
  }

  #[test]
  fn scan_node_cat_returns_best_when_no_exact() {
      // TI: [0,0,1], de=[2], DSBC → no exact match → best is any (all d=1.0) → first scanned
      let lo = array![0_i64];
      let win_shape = array![3_i64];
      let ti_flat = array![0.0_f64, 0.0_f64, 1.0_f64];
      let ti_strides = array![1_i64];
      let de_sim = array![2.0_f64];
      let lag_flat = array![0_i64];
      let weights = array![1.0_f64];
      let result = scan_node_categorical(
          lo.view(), win_shape.view(), 0, 3, 0.0,
          de_sim.view(), ti_flat.view(), ti_strides.view(),
          lag_flat.view(), weights.view(),
      );
      assert!(result.is_some());
      assert_eq!(result.unwrap()[0], 0); // first scanned, all equal d=1.0
  }

  #[test]
  fn scan_node_cat_ds_mode_early_exit() {
      // DS mode: threshold=0.6. TI: [0,1,2], de=[1], d_max is for categorical so weights=[1.0]
      // candidate 0: ti[0]=0 ≠ 1 → d=1.0 (≥0.6, no exit); candidate 1: ti[1]=1 == 1 → d=0.0 < 0.6 → exit
      let lo = array![0_i64];
      let win_shape = array![3_i64];
      let ti_flat = array![0.0_f64, 1.0_f64, 2.0_f64];
      let ti_strides = array![1_i64];
      let de_sim = array![1.0_f64];
      let lag_flat = array![0_i64];
      let weights = array![1.0_f64];
      let result = scan_node_categorical(
          lo.view(), win_shape.view(), 0, 3, 0.6,
          de_sim.view(), ti_flat.view(), ti_strides.view(),
          lag_flat.view(), weights.view(),
      );
      assert!(result.is_some());
      assert_eq!(result.unwrap()[0], 1);
  }
  ```

- [ ] **Step 2: Implement `scan_node_categorical` in `src/mps.rs`**

  ```rust
  /// Full DSBC/DS scan for one simulation node — univariate categorical case.
  ///
  /// Combines the Python `_scan_window` outer loop with `_dist_block` into a single
  /// Rust function. Iterates over candidate anchors in scan order (random start,
  /// sequential wrap-around) and computes the weighted categorical distance for each.
  /// Returns the TI coordinates of the best match, or `None` if every candidate has
  /// distance `f64::INFINITY` (only reachable on a masked TI — not triggered when
  /// `ti_has_nan = False`, which is required to reach this function).
  ///
  /// # Arguments
  ///
  /// * `lo` – Lower-left anchor of the TI search window (TI coordinates), shape `(dim,)`.
  /// * `win_shape` – Search window shape, shape `(dim,)`.
  /// * `start` – Flat index into the window to start scanning from.
  ///   Computed in Python as `int(u_start_i * win_size)`.
  /// * `max_scan` – Maximum candidates to evaluate.
  /// * `threshold` – DSBC: `<= 0` (accept first exact match `d == 0`);
  ///   DS: `> 0` (accept first `d < threshold`).
  /// * `de_sim` – Data-event categorical values at the sim node, shape `(n,)`.
  /// * `ti_flat` – Flattened TI (categories as `f64`), shape `(ti_size,)`.
  /// * `ti_strides` – C-order strides into the TI, shape `(dim,)`.
  /// * `lag_flat` – Flat lag offsets, shape `(n,)`.
  /// * `node_weights` – Precomputed weights, shape `(n,)`.
  ///
  /// # Returns
  ///
  /// `Some(coords)` where `coords` has shape `(dim,)` with the TI coordinates of the
  /// best-match anchor, or `None` if all candidates have `f64::INFINITY` distance.
  pub fn scan_node_categorical(
      lo: ArrayView1<'_, i64>,
      win_shape: ArrayView1<'_, i64>,
      start: usize,
      max_scan: usize,
      threshold: f64,
      de_sim: ArrayView1<'_, f64>,
      ti_flat: ArrayView1<'_, f64>,
      ti_strides: ArrayView1<'_, i64>,
      lag_flat: ArrayView1<'_, i64>,
      node_weights: ArrayView1<'_, f64>,
  ) -> Option<Array1<i64>> {
      let dim = lo.len();
      let win_shape_u: Vec<usize> = win_shape.iter().map(|&s| s as usize).collect();
      let win_size: usize = win_shape_u.iter().product();

      // C-order strides into the window (not the TI)
      let mut win_strides = vec![1usize; dim];
      for d in (0..dim - 1).rev() {
          win_strides[d] = win_strides[d + 1] * win_shape_u[d + 1];
      }

      let dsbc = threshold <= 0.0;
      let mut best_d = f64::INFINITY;
      let mut best_coords: Option<Array1<i64>> = None;

      for k in 0..max_scan {
          let pos = (start + k) % win_size;
          // Decode flat window position into dim-D TI coordinates
          let mut coords = Array1::<i64>::zeros(dim);
          let mut rem = pos;
          for d in 0..dim {
              let c = (rem / win_strides[d]) as i64;
              coords[d] = lo[d] + c;
              rem %= win_strides[d];
          }
          // Flat base index in TI for this candidate anchor
          let base: i64 = coords.iter().zip(ti_strides.iter()).map(|(&c, &s)| c * s).sum();

          // Weighted categorical distance for this candidate
          let d = de_sim
              .iter()
              .zip(lag_flat.iter())
              .zip(node_weights.iter())
              .fold(0.0_f64, |acc, ((&de_val, &lag), &wt)| {
                  let ti_val = ti_flat[(base + lag) as usize];
                  acc + if (de_val - ti_val).abs() > 0.5 { wt } else { 0.0 }
              });

          // DSBC: accept exact match immediately; DS: accept first under threshold
          if (dsbc && d == 0.0) || (!dsbc && d < threshold) {
              return Some(coords);
          }
          if d < best_d {
              best_d = d;
              best_coords = Some(coords);
          }
      }
      best_coords
  }
  ```

- [ ] **Step 3: Run tests**

  ```bash
  cargo test mps::tests -- --nocapture
  ```

  Expected: 13 tests pass.

- [ ] **Step 4: Add `#[pyfunction]` wrapper in `src/lib.rs`**

  Add import inside `mod gstools_core { ... }`:

  ```rust
  use crate::mps::scan_node_categorical;
  ```

  Add PyO3 wrapper:

  ```rust
  #[pyfunction(name = "mps_scan_node_cat")]
  #[allow(clippy::too_many_arguments)]
  fn mps_scan_node_cat_py<'py>(
      py: Python<'py>,
      lo: PyReadonlyArray1<i64>,
      win_shape: PyReadonlyArray1<i64>,
      start: usize,
      max_scan: usize,
      threshold: f64,
      de_sim: PyReadonlyArray1<f64>,
      ti_flat: PyReadonlyArray1<f64>,
      ti_strides: PyReadonlyArray1<i64>,
      lag_flat: PyReadonlyArray1<i64>,
      node_weights: PyReadonlyArray1<f64>,
  ) -> Option<Bound<'py, PyArray1<i64>>> {
      scan_node_categorical(
          lo.as_array(),
          win_shape.as_array(),
          start,
          max_scan,
          threshold,
          de_sim.as_array(),
          ti_flat.as_array(),
          ti_strides.as_array(),
          lag_flat.as_array(),
          node_weights.as_array(),
      )
      .map(|arr| arr.into_pyarray(py))
  }
  ```

- [ ] **Step 5: Build and add Rust dispatch in `scan.py`**

  Build:

  ```bash
  cargo clippy --all-targets --all-features -- -D warnings
  env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV /Users/lealroja/Documents/UFZ/RUST/.venv/bin/maturin develop --release
  ```

  In `src/gstools/mps/scan.py`, add import at module level (inside the `if _mps_config._GSTOOLS_CORE_AVAIL:` block):

  ```python
  from gstools_core import mps_scan_node_cat as _mps_scan_node_cat_gsc
  ```

  In `_scan_for_match`, before the `return _scan_window(...)` call at the bottom, add a Rust fast-path for the univariate categorical case:

  ```python
  # Phase 2: full node scan in Rust for univariate categorical (no NaN TI)
  if (
      _use_rust
      and len(active_vars) == 1
      and cfg.var_categorical[active_vars[0]]
  ):
      v = active_vars[0]
      result = _mps_scan_node_cat_gsc(
          np.asarray(lo, dtype=np.int64),
          np.asarray(win_shape, dtype=np.int64),
          start,
          max_scan,
          cfg.threshold,
          de_v[v],               # float64, SG values
          cfg.ti_flat_f64[v],    # float64 pre-cast TI
          cfg.ti_strides[v].astype(np.int64),
          flat_il[v].astype(np.int64),
          precomp_w[v],
      )
      return result  # ndarray shape (dim,) or None

  return _scan_window(lo, win_shape, start, max_scan, cfg.threshold, _dist_block)
  ```

- [ ] **Step 6: Add Phase 2 regression test**

  Add to `tests/test_mps_rust_backend.py`:

  ```python
  def test_phase2_scan_node_categorical():
      """Phase 2 full-node scan produces bit-identical output on a larger grid."""
      ti = _cat_ti((120, 120))
      py = _run(ti, True, "l1", (40, 40), 0.3, 0.0, 77, False)
      rs = _run(ti, True, "l1", (40, 40), 0.3, 0.0, 77, True)
      np.testing.assert_array_equal(py, rs)
  ```

- [ ] **Step 7: Run all regression tests**

  ```bash
  PYTHONPATH=src /Users/lealroja/miniconda3/envs/gstools/bin/python -m pytest tests/test_mps_rust_backend.py -v
  ```

  Expected: 15 tests pass.

- [ ] **Step 8: Commit both repos**

  ```bash
  cd /Users/lealroja/Documents/UFZ/MPS-Tools/GSTools-Core
  git add src/mps.rs src/lib.rs
  git commit -m "feat(mps): Phase 2 — scan_node_categorical full DSBC/DS scan in Rust"

  cd /Users/lealroja/Documents/UFZ/MPS-Tools/GSTools
  git add src/gstools/mps/scan.py tests/test_mps_rust_backend.py
  git commit -m "feat(mps): Phase 2 — bypass Python _scan_window for univariate categorical"
  ```

---

## Task 9 (Phase 3): `select_neighbors` in Rust

*Defer until Phase 1+2 benchmarks show Phase 3 worth the effort (5.8 % at field scale). Check benchmark output from Task 7 first.*

**Files:**
- Modify: `src/mps.rs`
- Modify: `src/lib.rs`
- Modify: `src/gstools/mps/neighbors.py`

**Goal:** Replace the chunked Python `for`-loop in `_select_neighbors` (neighbors.py:110–157) with a Rust function. The function receives the pre-sorted offset array and a flat `path_pos_map` and returns the closest `n_neighbors` valid neighbor coordinates + path indices.

**Interfaces:**
- Produces: `pub fn select_neighbors(x_i, offset_arr, sim_shape, path_pos_map, curr_idx, informed_flat, max_radius_sq, n_neighbors) -> (Array2<i64>, Array1<i64>)`
- Python exports as: `mps_select_neighbors`
- Dispatch in `neighbors.py`: guard `_use_rust and informed is None` (the informed-is-None case covers the serial path where the check is `vi < curr_idx or vi == -1`).

*(Full implementation code follows the same TDD pattern as Tasks 2–3. Omitted here to keep plan length manageable; implement after Phase 1+2 is stable.)*

---

## Self-Review Checklist

**Spec coverage:**

| Requirement | Task(s) |
|---|---|
| Phase 1a `_dist_block` replaced | Tasks 2, 4, 5 |
| Phase 1b `vec_categorical_dist` absorbed into Phase 1a | Tasks 2, 4 |
| Continuous distances L1/L2/Lp | Tasks 3, 4, 5 |
| Variation distance falls through to Python | Task 6 test `test_variation_distance_falls_back_to_python` |
| Python fallback preserved (no gstools_core) | Task 5 Step 5 |
| Bit-identical categorical output | Task 5 Step 6, Task 6 |
| Float-close continuous output | Task 6 tests |
| Conditioning data preserved | Task 6 `test_conditioned_categorical` |
| Phase 2 `_scan_window` replaced | Task 8 |
| Phase 3 `_select_neighbors` | Task 9 (deferred) |
| ASV BACKENDS benchmark | Task 7 |
| Markdown speedup report (Python vs Rust per case) | Task 10 |
| Dispatch flags match existing pattern | Task 5 Step 2 (`_GSTOOLS_CORE_AVAIL` import guard) |

**Placeholder scan:** None. Tasks 1–8 contain complete code. Task 9 intentionally deferred with explicit rationale.

**Type consistency across tasks:**

| Array | Python dtype | Rust type | Cast needed |
|---|---|---|---|
| `de_v[v]` (data event) | `float64` | `ArrayView1<f64>` | None (SG always float64) |
| `cfg.ti_flat_f64[v]` | `float64` (pre-cast) | `ArrayView1<f64>` | Pre-cast in `simulate.py` |
| `base` (`y_blk @ ti_strides`) | `intp` = `int64` | `ArrayView1<i64>` | `.astype(np.int64)` (no-op on 64-bit) |
| `flat_il[v]` | `float64` (integer values) | `ArrayView1<i64>` | `.astype(np.int64)` |
| `precomp_w[v]` | `float64` | `ArrayView1<f64>` | None |
| Return `dist_block_*` | — | `Array1<f64>` → `PyArray1<f64>` | `into_pyarray(py)` |
| Return `scan_node_cat` | — | `Option<Array1<i64>>` → `Option<PyArray1<i64>>` | `.map(into_pyarray)` |
