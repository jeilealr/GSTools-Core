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
}
