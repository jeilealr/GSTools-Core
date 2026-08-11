//! MPS (Multiple Point Statistics) kernels for Direct Sampling.
//!
//! Fused gather + distance functions that replace the Python `_dist_block`
//! closure in `gstools/mps/scan.py`. Each kernel reads TI candidates via
//! flat-index arithmetic and computes weighted distances in one pass,
//! avoiding the temporary `all_de_ti` allocation of the Python path.

use ndarray::{Array1, ArrayView1};

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
/// * `de_sim` – Data-event category labels at the current simulation node, shape `(n,)`.
///   Labels are stored as `f64` (the same representation as the simulation grid).
/// * `ti_flat` – Flattened C-order TI array, shape `(ti_size,)`.
///   Category labels are stored as `f64` (pre-cast by `_DirectSamplingEngine`).
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
                // Match NumPy's categorical comparison exactly: any distinct label
                // is a mismatch, including non-integer labels close together.
                d + if de_val != ti_val { wt } else { 0.0 }
            })
    }))
}

/// Fused gather + masked weighted categorical distance over TI candidates.
///
/// This is the masked counterpart of [`dist_block_categorical`]. For every
/// candidate, TI event positions whose value is `NaN` are excluded. The
/// surviving node weights are renormalized within that candidate:
///
/// `dist[b] = Σ(valid mismatch weights) / Σ(valid weights)`
///
/// A candidate with no positive surviving weight returns `+∞`, matching
/// Python's `vec_categorical_dist(..., has_nan=True)`. Missingness is
/// intentionally defined by `is_nan`, not `is_finite`: infinities remain
/// ordinary numeric category labels in event-distance calculations.
///
/// The caller guarantees all flat indices are in bounds. Python continues to
/// validate candidate centers, select the winner, handle fallback, retrieve
/// the selected value, and write the simulation grid.
///
/// # Arguments
///
/// * `de_sim` – Data-event category labels, shape `(n,)`.
/// * `ti_flat` – Flattened C-order TI, including possible `NaN`, shape `(ti_size,)`.
/// * `base_flat` – Flat candidate-anchor indices, shape `(B,)`.
/// * `lag_flat` – Flat event-lag offsets, shape `(n,)`.
/// * `node_weights` – Precomputed node weights, shape `(n,)`.
///
/// # Returns
///
/// Distance array of shape `(B,)`; excluded candidates contain `+∞`.
pub fn dist_block_categorical_masked(
    de_sim: ArrayView1<'_, f64>,
    ti_flat: ArrayView1<'_, f64>,
    base_flat: ArrayView1<'_, i64>,
    lag_flat: ArrayView1<'_, i64>,
    node_weights: ArrayView1<'_, f64>,
) -> Array1<f64> {
    Array1::from_iter(base_flat.iter().map(|&base| {
        let (mismatch, weight_sum) = de_sim
            .iter()
            .zip(lag_flat.iter())
            .zip(node_weights.iter())
            .fold(
                (0.0_f64, 0.0_f64),
                |(mismatch, weight_sum), ((&de_val, &lag), &weight)| {
                    let ti_val = ti_flat[(base + lag) as usize];
                    if ti_val.is_nan() {
                        (mismatch, weight_sum)
                    } else {
                        (
                            mismatch + if de_val != ti_val { weight } else { 0.0 },
                            weight_sum + weight,
                        )
                    }
                },
            );
        if weight_sum > 0.0 {
            mismatch / weight_sum
        } else {
            f64::INFINITY
        }
    }))
}

/// Fused gather + weighted L1 distance over a block of TI candidates.
///
/// `dist[b] = Σᵢ node_weights[i] · |de_sim[i] − ti_flat[base_flat[b] + lag_flat[i]]| / d_max`
///
/// # Arguments
///
/// * `de_sim` – Data-event values at the current simulation node, shape `(n,)`.
/// * `ti_flat` – Flattened C-order TI array, shape `(ti_size,)`.
/// * `base_flat` – Flat base index for each candidate anchor, shape `(B,)`.
/// * `lag_flat` – Flat lag offsets, shape `(n,)`.
/// * `node_weights` – Precomputed spatial-decay weights normalized to sum 1, shape `(n,)`.
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

/// Fused gather + masked weighted L1 distance over TI candidates.
///
/// This preserves `vec_l1_dist(..., has_nan=True)`: candidate TI event
/// positions containing `NaN` are excluded, surviving node weights are
/// renormalized per candidate, and a candidate with no positive surviving
/// weight returns `+∞`.
///
/// `dist[b] = Σ(valid weight · |SG − TI| / d_max) / Σ(valid weights)`
///
/// Missingness is deliberately `is_nan`, not `is_finite`, to match the current
/// Python oracle exactly. The existing [`dist_block_l1`] remains the separate,
/// overhead-free NaN-free path.
///
/// # Arguments
///
/// * `de_sim` – Simulation-grid event values, shape `(n,)`.
/// * `ti_flat` – Flattened C-order TI, including possible `NaN`, shape `(ti_size,)`.
/// * `base_flat` – Flat candidate-anchor indices, shape `(B,)`.
/// * `lag_flat` – Flat event-lag offsets, shape `(n,)`.
/// * `node_weights` – Precomputed node weights, shape `(n,)`.
/// * `d_max` – TI finite-value range used for normalization. Must be positive.
///
/// # Returns
///
/// Distance array of shape `(B,)`; excluded candidates contain `+∞`.
pub fn dist_block_l1_masked(
    de_sim: ArrayView1<'_, f64>,
    ti_flat: ArrayView1<'_, f64>,
    base_flat: ArrayView1<'_, i64>,
    lag_flat: ArrayView1<'_, i64>,
    node_weights: ArrayView1<'_, f64>,
    d_max: f64,
) -> Array1<f64> {
    Array1::from_iter(base_flat.iter().map(|&base| {
        let (error, weight_sum) = de_sim
            .iter()
            .zip(lag_flat.iter())
            .zip(node_weights.iter())
            .fold(
                (0.0_f64, 0.0_f64),
                |(error, weight_sum), ((&de_val, &lag), &weight)| {
                    let ti_val = ti_flat[(base + lag) as usize];
                    if ti_val.is_nan() {
                        (error, weight_sum)
                    } else {
                        (
                            error + weight * (de_val - ti_val).abs() / d_max,
                            weight_sum + weight,
                        )
                    }
                },
            );
        if weight_sum > 0.0 {
            error / weight_sum
        } else {
            f64::INFINITY
        }
    }))
}

/// Fused gather + weighted L2 distance over a block of TI candidates.
///
/// `dist[b] = √(Σᵢ node_weights[i] · ((de_sim[i] − ti_flat[base_flat[b] + lag_flat[i]]) / d_max)²)`
///
/// # Arguments
///
/// * `de_sim` – Data-event values at the current simulation node, shape `(n,)`.
/// * `ti_flat` – Flattened C-order TI array, shape `(ti_size,)`.
/// * `base_flat` – Flat base index for each candidate anchor, shape `(B,)`.
/// * `lag_flat` – Flat lag offsets, shape `(n,)`.
/// * `node_weights` – Precomputed spatial-decay weights normalized to sum 1, shape `(n,)`.
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
/// * `de_sim` – Data-event values at the current simulation node, shape `(n,)`.
/// * `ti_flat` – Flattened C-order TI array, shape `(ti_size,)`.
/// * `base_flat` – Flat base index for each candidate anchor, shape `(B,)`.
/// * `lag_flat` – Flat lag offsets, shape `(n,)`.
/// * `node_weights` – Precomputed spatial-decay weights normalized to sum 1, shape `(n,)`.
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

/// Fused gather + weighted variation distance over a block of TI candidates.
///
/// Variation distance compares the *shape* of the pattern (deviation from the
/// local mean) rather than raw values, so a constant vertical shift between two
/// otherwise identical patterns has distance zero. For each candidate `b`, with
/// SG mean `S̄ = mean(de_sim)` and candidate TI mean `T̄_b = mean over the event
/// positions of that candidate`:
///
/// `dist[b] = min(1, [Σᵢ node_weights[i] · |((de_sim[i] − S̄) − (T_{b,i} − T̄_b)) / (2·d_max)|ᵖ]^(1/p))`
///
/// Both means are **unweighted** arithmetic means over the `n` event positions;
/// the node weights apply only to the centered error contributions. The `S̄`
/// mean is computed once per call; `T̄_b` is candidate-specific and needs one
/// extra pass over the event positions per candidate.
///
/// This kernel is the NaN-free counterpart of `vec_variation_dist` in
/// `gstools/mps/distance.py`. It returns only the distance vector — the
/// winning-anchor mean-shift assignment (Mariethoz2010 Eq. 9) stays in Python
/// (`TrainingImage.adjust_value`), so Rust does not return an adjusted value.
///
/// # Arguments
///
/// * `de_sim` – Data-event values at the current simulation node, shape `(n,)`.
/// * `ti_flat` – Flattened C-order TI array, shape `(ti_size,)`.
/// * `base_flat` – Flat base index for each candidate anchor, shape `(B,)`.
/// * `lag_flat` – Flat lag offsets, shape `(n,)`.
/// * `node_weights` – Precomputed spatial-decay weights normalized to sum 1, shape `(n,)`.
/// * `d_max` – TI value range for normalization. Must be positive.
/// * `p` – Variation Lp exponent; from `Variable.variation_p_norm`. Must be positive.
///
/// # Returns
///
/// Distance array of shape `(B,)` with values in `[0, 1]`.
#[allow(clippy::too_many_arguments)]
pub fn dist_block_variation(
    de_sim: ArrayView1<'_, f64>,
    ti_flat: ArrayView1<'_, f64>,
    base_flat: ArrayView1<'_, i64>,
    lag_flat: ArrayView1<'_, i64>,
    node_weights: ArrayView1<'_, f64>,
    d_max: f64,
    p: f64,
) -> Array1<f64> {
    let n = de_sim.len() as f64;
    // SG mean is the same for every candidate; compute it once.
    let de_mean = de_sim.sum() / n;
    let two_dmax = 2.0 * d_max;
    // `f64::min` ignores a NaN operand, whereas NumPy's `minimum` propagates
    // it. Preserve the Python oracle for non-finite direct-FFI inputs too.
    let clamp = |value: f64| {
        if value.is_nan() {
            value
        } else {
            value.min(1.0)
        }
    };

    // Candidate TI mean over the event positions (unweighted); one pass.
    let ti_mean = |base: i64| -> f64 {
        lag_flat
            .iter()
            .map(|&lag| ti_flat[(base + lag) as usize])
            .sum::<f64>()
            / n
    };

    // Specialize p == 2 (the `variation` default) and p == 1 to avoid a
    // per-element `powf`, which otherwise dominates runtime and makes the
    // fused kernel slower than NumPy's vectorized power. General `p` keeps the
    // `powf` path. `2*d_max` normalizes the common case to [0, 1]; SG values
    // are not bounded by the TI range (conditioning / mean-shifts), so clamp.
    if p == 2.0 {
        Array1::from_iter(base_flat.iter().map(|&base| {
            let tm = ti_mean(base);
            let inner = de_sim
                .iter()
                .zip(lag_flat.iter())
                .zip(node_weights.iter())
                .fold(0.0_f64, |acc, ((&de_val, &lag), &wt)| {
                    let ti_val = ti_flat[(base + lag) as usize];
                    let c = ((de_val - de_mean) - (ti_val - tm)) / two_dmax;
                    acc + wt * c * c
                });
            clamp(inner.sqrt())
        }))
    } else if p == 1.0 {
        Array1::from_iter(base_flat.iter().map(|&base| {
            let tm = ti_mean(base);
            let inner = de_sim
                .iter()
                .zip(lag_flat.iter())
                .zip(node_weights.iter())
                .fold(0.0_f64, |acc, ((&de_val, &lag), &wt)| {
                    let ti_val = ti_flat[(base + lag) as usize];
                    let c = ((de_val - de_mean) - (ti_val - tm)) / two_dmax;
                    acc + wt * c.abs()
                });
            clamp(inner)
        }))
    } else {
        let inv_p = p.recip();
        Array1::from_iter(base_flat.iter().map(|&base| {
            let tm = ti_mean(base);
            let inner = de_sim
                .iter()
                .zip(lag_flat.iter())
                .zip(node_weights.iter())
                .fold(0.0_f64, |acc, ((&de_val, &lag), &wt)| {
                    let ti_val = ti_flat[(base + lag) as usize];
                    let diff = (de_val - de_mean) - (ti_val - tm);
                    acc + wt * (diff / two_dmax).abs().powf(p)
                });
            clamp(inner.powf(inv_p))
        }))
    }
}

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
#[allow(clippy::too_many_arguments)]
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
        let base: i64 = coords
            .iter()
            .zip(ti_strides.iter())
            .map(|(&c, &s)| c * s)
            .sum();

        // Weighted categorical distance for this candidate
        let d = de_sim
            .iter()
            .zip(lag_flat.iter())
            .zip(node_weights.iter())
            .fold(0.0_f64, |acc, ((&de_val, &lag), &wt)| {
                let ti_val = ti_flat[(base + lag) as usize];
                acc + if de_val != ti_val { wt } else { 0.0 }
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
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
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
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
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
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
        );
        assert_abs_diff_eq!(d[0], 0.5, epsilon = 1e-12);
    }

    #[test]
    fn categorical_close_distinct_labels_are_mismatch() {
        // Categorical labels need not be integer-spaced: 0.1 and 0.2 are distinct.
        let de_sim = array![0.1_f64];
        let ti_flat = array![0.2_f64];
        let base_flat = array![0_i64];
        let lag_flat = array![0_i64];
        let weights = array![1.0_f64];
        let d = dist_block_categorical(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
        );
        assert_abs_diff_eq!(d[0], 1.0, epsilon = 1e-12);
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
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
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
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
        );
        assert_abs_diff_eq!(d[0], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn categorical_masked_renormalizes_each_candidate() {
        let de_sim = array![0.1_f64, 0.2_f64, 0.3_f64];
        let ti_flat = array![
            0.1_f64,
            f64::NAN,
            0.4_f64,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            0.2_f64,
            0.2_f64,
            0.3_f64,
        ];
        let base_flat = array![0_i64, 3_i64, 6_i64];
        let lag_flat = array![0_i64, 1_i64, 2_i64];
        let weights = array![0.2_f64, 0.3_f64, 0.5_f64];
        let d = dist_block_categorical_masked(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
        );
        // Candidate 0: only positions 0 and 2 survive. Position 2 mismatches,
        // so distance = 0.5 / (0.2 + 0.5) = 5/7.
        assert_abs_diff_eq!(d[0], 5.0 / 7.0, epsilon = 1e-12);
        assert!(d[1].is_infinite());
        // Candidate 2 differs only at the first close numeric label.
        assert_abs_diff_eq!(d[2], 0.2, epsilon = 1e-12);
    }

    #[test]
    fn categorical_masked_zero_surviving_weight_is_infinite() {
        let de_sim = array![1.0_f64, 1.0_f64];
        let ti_flat = array![2.0_f64, f64::NAN];
        let base_flat = array![0_i64];
        let lag_flat = array![0_i64, 1_i64];
        let weights = array![0.0_f64, 1.0_f64];
        let d = dist_block_categorical_masked(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
        );
        assert!(d[0].is_infinite());
    }

    #[test]
    fn categorical_masked_matches_unmasked_without_nan() {
        let de_sim = array![0.1_f64, 0.2_f64, 0.3_f64];
        let ti_flat = array![0.2_f64, 0.2_f64, 0.4_f64];
        let base_flat = array![0_i64];
        let lag_flat = array![0_i64, 1_i64, 2_i64];
        let weights = array![0.25_f64, 0.25_f64, 0.5_f64];
        let plain = dist_block_categorical(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
        );
        let masked = dist_block_categorical_masked(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
        );
        assert_abs_diff_eq!(masked[0], plain[0], epsilon = 1e-12);
    }

    #[test]
    fn l1_single_lag() {
        // |5.0 - 3.0| / 4.0 * 1.0 = 0.5
        let de_sim = array![5.0_f64];
        let ti_flat = array![3.0_f64];
        let base_flat = array![0_i64];
        let lag_flat = array![0_i64];
        let weights = array![1.0_f64];
        let d = dist_block_l1(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            4.0,
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
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            4.0,
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
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            4.0,
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
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            4.0,
        );
        let d_lp = dist_block_lp(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            4.0,
            1.0,
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
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            4.0,
        );
        let d_lp = dist_block_lp(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            4.0,
            2.0,
        );
        assert_abs_diff_eq!(d_l2[0], d_lp[0], epsilon = 1e-10);
    }

    #[test]
    fn variation_identical_pattern_zero() {
        // de == ti exactly → centered diffs are 0 → distance 0.
        let de_sim = array![0.0_f64, 1.0_f64, 2.0_f64, 1.0_f64];
        let ti_flat = array![0.0_f64, 1.0_f64, 2.0_f64, 1.0_f64];
        let base_flat = array![0_i64];
        let lag_flat = array![0_i64, 1_i64, 2_i64, 3_i64];
        let weights = array![0.25_f64, 0.25_f64, 0.25_f64, 0.25_f64];
        let d = dist_block_variation(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            2.0,
            2.0,
        );
        assert_abs_diff_eq!(d[0], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn variation_constant_shift_zero() {
        // ti = de + 5 (constant vertical shift) → variation distance 0.
        let de_sim = array![1.0_f64, 2.0_f64, 3.0_f64];
        let ti_flat = array![6.0_f64, 7.0_f64, 8.0_f64];
        let base_flat = array![0_i64];
        let lag_flat = array![0_i64, 1_i64, 2_i64];
        let weights = array![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
        let d = dist_block_variation(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            2.0,
            2.0,
        );
        assert_abs_diff_eq!(d[0], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn variation_p1_hand_calc() {
        // de=[0,2] mean=1 → de_c=[-1,1]; ti=[0,4] mean=2 → ti_c=[-2,2].
        // diff = de_c - ti_c = [1,-1]; d_max=1 → 2*d_max=2; |diff/2|=[0.5,0.5].
        // p=1, w=[0.5,0.5]: inner = 0.5*0.5 + 0.5*0.5 = 0.5 → d = 0.5.
        let de_sim = array![0.0_f64, 2.0_f64];
        let ti_flat = array![0.0_f64, 4.0_f64];
        let base_flat = array![0_i64];
        let lag_flat = array![0_i64, 1_i64];
        let weights = array![0.5_f64, 0.5_f64];
        let d = dist_block_variation(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            1.0,
            1.0,
        );
        assert_abs_diff_eq!(d[0], 0.5, epsilon = 1e-12);
    }

    #[test]
    fn variation_p2_hand_calc() {
        // Same inputs as p1 case but p=2:
        // inner = 0.5*0.5^2 + 0.5*0.5^2 = 0.25 → d = sqrt(0.25) = 0.5.
        let de_sim = array![0.0_f64, 2.0_f64];
        let ti_flat = array![0.0_f64, 4.0_f64];
        let base_flat = array![0_i64];
        let lag_flat = array![0_i64, 1_i64];
        let weights = array![0.5_f64, 0.5_f64];
        let d = dist_block_variation(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            1.0,
            2.0,
        );
        assert_abs_diff_eq!(d[0], 0.5, epsilon = 1e-12);
    }

    #[test]
    fn variation_nonuniform_weights() {
        // de=[0,3] mean=1.5 → de_c=[-1.5,1.5]; ti=[0,1] mean=0.5 → ti_c=[-0.5,0.5].
        // diff=[-1,1]; d_max=1 → 2*d_max=2; |diff/2|=[0.5,0.5]; p=1.
        // w=[0.25,0.75]: inner = 0.25*0.5 + 0.75*0.5 = 0.5 → d = 0.5.
        let de_sim = array![0.0_f64, 3.0_f64];
        let ti_flat = array![0.0_f64, 1.0_f64];
        let base_flat = array![0_i64];
        let lag_flat = array![0_i64, 1_i64];
        let weights = array![0.25_f64, 0.75_f64];
        let d = dist_block_variation(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            1.0,
            1.0,
        );
        assert_abs_diff_eq!(d[0], 0.5, epsilon = 1e-12);
    }

    #[test]
    fn variation_clamps_to_one() {
        // de=[0,10] mean=5 → de_c=[-5,5]; ti=[0,0] mean=0 → ti_c=[0,0].
        // diff=[-5,5]; d_max=1 → |diff/2|=[2.5,2.5]; p=1, w=[0.5,0.5]:
        // inner = 2.5 → raw d = 2.5, clamped to 1.0.
        let de_sim = array![0.0_f64, 10.0_f64];
        let ti_flat = array![0.0_f64, 0.0_f64];
        let base_flat = array![0_i64];
        let lag_flat = array![0_i64, 1_i64];
        let weights = array![0.5_f64, 0.5_f64];
        let d = dist_block_variation(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            1.0,
            1.0,
        );
        assert_abs_diff_eq!(d[0], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn variation_two_candidates() {
        // TI=[0,2,0,4], de=[0,2] mean=1. d_max=1, p=1, w=[0.5,0.5].
        // base 0, lags=[0,1]: ti=[0,2] mean=1 → ti_c=[-1,1]; de_c=[-1,1]; diff=0 → d=0.
        // base 2, lags=[0,1]: ti=[0,4] mean=2 → ti_c=[-2,2]; diff=[1,-1]; |.|/2=0.5 → d=0.5.
        let de_sim = array![0.0_f64, 2.0_f64];
        let ti_flat = array![0.0_f64, 2.0_f64, 0.0_f64, 4.0_f64];
        let base_flat = array![0_i64, 2_i64];
        let lag_flat = array![0_i64, 1_i64];
        let weights = array![0.5_f64, 0.5_f64];
        let d = dist_block_variation(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            1.0,
            1.0,
        );
        assert_abs_diff_eq!(d[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(d[1], 0.5, epsilon = 1e-12);
    }

    #[test]
    fn l1_masked_renormalizes_each_candidate() {
        let de_sim = array![0.0_f64, 10.0_f64, 20.0_f64];
        let ti_flat = array![
            2.0_f64,
            f64::NAN,
            10.0_f64,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            0.0_f64,
            0.0_f64,
            20.0_f64,
        ];
        let base_flat = array![0_i64, 3_i64, 6_i64];
        let lag_flat = array![0_i64, 1_i64, 2_i64];
        let weights = array![0.2_f64, 0.3_f64, 0.5_f64];
        let d = dist_block_l1_masked(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            20.0,
        );
        // Candidate 0 numerator = 0.2*(2/20) + 0.5*(10/20) = 0.27;
        // surviving weight = 0.7.
        assert_abs_diff_eq!(d[0], 0.27 / 0.7, epsilon = 1e-12);
        assert!(d[1].is_infinite());
        assert_abs_diff_eq!(d[2], 0.15, epsilon = 1e-12);
    }

    #[test]
    fn l1_masked_zero_surviving_weight_is_infinite() {
        let de_sim = array![1.0_f64, 1.0_f64];
        let ti_flat = array![2.0_f64, f64::NAN];
        let base_flat = array![0_i64];
        let lag_flat = array![0_i64, 1_i64];
        let weights = array![0.0_f64, 1.0_f64];
        let d = dist_block_l1_masked(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            1.0,
        );
        assert!(d[0].is_infinite());
    }

    #[test]
    fn l1_masked_matches_unmasked_without_nan() {
        let de_sim = array![0.0_f64, 10.0_f64, 20.0_f64];
        let ti_flat = array![2.0_f64, 4.0_f64, 10.0_f64];
        let base_flat = array![0_i64];
        let lag_flat = array![0_i64, 1_i64, 2_i64];
        let weights = array![0.25_f64, 0.25_f64, 0.5_f64];
        let plain = dist_block_l1(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            20.0,
        );
        let masked = dist_block_l1_masked(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            20.0,
        );
        assert_abs_diff_eq!(masked[0], plain[0], epsilon = 1e-12);
    }

    #[test]
    fn variation_nonfinite_input_propagates_nan_like_numpy() {
        // A TI event containing infinity has an infinite mean; centering then
        // produces NaN. NumPy propagates that NaN through its final clamp.
        let de_sim = array![0.0_f64, 1.0_f64];
        let ti_flat = array![0.0_f64, f64::INFINITY];
        let base_flat = array![0_i64];
        let lag_flat = array![0_i64, 1_i64];
        let weights = array![0.5_f64, 0.5_f64];
        let d = dist_block_variation(
            de_sim.view(),
            ti_flat.view(),
            base_flat.view(),
            lag_flat.view(),
            weights.view(),
            1.0,
            2.0,
        );
        assert!(d[0].is_nan());
    }

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
            lo.view(),
            win_shape.view(),
            0,
            5,
            0.0,
            de_sim.view(),
            ti_flat.view(),
            ti_strides.view(),
            lag_flat.view(),
            weights.view(),
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
            lo.view(),
            win_shape.view(),
            0,
            3,
            0.0,
            de_sim.view(),
            ti_flat.view(),
            ti_strides.view(),
            lag_flat.view(),
            weights.view(),
        );
        assert!(result.is_some());
        assert_eq!(result.unwrap()[0], 0); // first scanned, all equal d=1.0
    }

    #[test]
    fn scan_node_cat_ds_mode_early_exit() {
        // DS mode: threshold=0.6. TI: [0,1,2], de=[1]
        // candidate 0: ti[0]=0 ≠ 1 → d=1.0 (≥0.6, no exit); candidate 1: ti[1]=1 == 1 → d=0.0 < 0.6 → exit
        let lo = array![0_i64];
        let win_shape = array![3_i64];
        let ti_flat = array![0.0_f64, 1.0_f64, 2.0_f64];
        let ti_strides = array![1_i64];
        let de_sim = array![1.0_f64];
        let lag_flat = array![0_i64];
        let weights = array![1.0_f64];
        let result = scan_node_categorical(
            lo.view(),
            win_shape.view(),
            0,
            3,
            0.6,
            de_sim.view(),
            ti_flat.view(),
            ti_strides.view(),
            lag_flat.view(),
            weights.view(),
        );
        assert!(result.is_some());
        assert_eq!(result.unwrap()[0], 1);
    }

    #[test]
    fn scan_node_cat_distinguishes_close_labels() {
        // Candidate 0 (0.2) differs from de=0.1; candidate 1 is the exact match.
        // The old tolerance-based comparison incorrectly accepted candidate 0.
        let lo = array![0_i64];
        let win_shape = array![2_i64];
        let ti_flat = array![0.2_f64, 0.1_f64];
        let ti_strides = array![1_i64];
        let de_sim = array![0.1_f64];
        let lag_flat = array![0_i64];
        let weights = array![1.0_f64];
        let result = scan_node_categorical(
            lo.view(),
            win_shape.view(),
            0,
            2,
            0.0,
            de_sim.view(),
            ti_flat.view(),
            ti_strides.view(),
            lag_flat.view(),
            weights.view(),
        );
        assert!(result.is_some());
        assert_eq!(result.unwrap()[0], 1);
    }
}
