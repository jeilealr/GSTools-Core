//! GSTools-Core
//!
//! `gstools_core` is a Rust implementation of the core algorithms of [GSTools].
//! At the moment, it is a drop in replacement for the Cython code included in GSTools.
//!
//! This crate includes
//! - [randomization methods](field) for the random field generation
//! - the [matrix operations](krige) of the kriging methods
//! - the [variogram estimation](variogram)
//!
//! [GSTools]: https://github.com/GeoStat-Framework/GSTools

#![warn(missing_docs)]

use pyo3::prelude::pymodule;

pub mod field;
pub mod krige;
pub mod mps;
pub mod mps_engine;
mod short_vec;
pub mod variogram;

#[pymodule]
mod gstools_core {
    use crate::field::{summator, summator_fourier, summator_incompr};
    use crate::krige::{calculator_field_krige, calculator_field_krige_and_variance};
    use crate::mps::{
        dist_block_categorical, dist_block_categorical_masked, dist_block_categorical_rayon,
        dist_block_l1, dist_block_l1_masked, dist_block_l2, dist_block_l2_masked, dist_block_lp,
        dist_block_lp_masked, dist_block_variation, dist_block_variation_masked, scan_node,
        scan_node_categorical,
    };
    use crate::mps_engine::simulate_engine;
    use crate::variogram::{
        variogram_directional, variogram_ma_structured, variogram_structured,
        variogram_unstructured,
    };
    use numpy::{
        IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
    };
    use pyo3::prelude::*;
    use pyo3::{exceptions::PyValueError, PyResult};

    type MpsSimulationPyOutput<'py> =
        (Bound<'py, PyArray2<f64>>, usize, usize, usize, usize, usize);

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add("__version__", env!("CARGO_PKG_VERSION"))?;
        Ok(())
    }

    #[pyfunction(name = "summate")]
    fn summate_py<'py>(
        py: Python<'py>,
        cov_samples: PyReadonlyArray2<f64>,
        z1: PyReadonlyArray1<f64>,
        z2: PyReadonlyArray1<f64>,
        pos: PyReadonlyArray2<f64>,
        num_threads: Option<usize>,
    ) -> Bound<'py, PyArray1<f64>> {
        let cov_samples = cov_samples.as_array();
        let z1 = z1.as_array();
        let z2 = z2.as_array();
        let pos = pos.as_array();
        summator(cov_samples, z1, z2, pos, num_threads).into_pyarray(py)
    }

    #[pyfunction(name = "summate_incompr")]
    fn summate_incompr_py<'py>(
        py: Python<'py>,
        cov_samples: PyReadonlyArray2<f64>,
        z1: PyReadonlyArray1<f64>,
        z2: PyReadonlyArray1<f64>,
        pos: PyReadonlyArray2<f64>,
        num_threads: Option<usize>,
    ) -> Bound<'py, PyArray2<f64>> {
        let cov_samples = cov_samples.as_array();
        let z1 = z1.as_array();
        let z2 = z2.as_array();
        let pos = pos.as_array();
        summator_incompr(cov_samples, z1, z2, pos, num_threads).into_pyarray(py)
    }

    #[pyfunction(name = "summate_fourier")]
    fn summate_fourier_py<'py>(
        py: Python<'py>,
        spectrum_factor: PyReadonlyArray1<f64>,
        modes: PyReadonlyArray2<f64>,
        z1: PyReadonlyArray1<f64>,
        z2: PyReadonlyArray1<f64>,
        pos: PyReadonlyArray2<f64>,
        num_threads: Option<usize>,
    ) -> Bound<'py, PyArray1<f64>> {
        let spectrum_factor = spectrum_factor.as_array();
        let modes = modes.as_array();
        let z1 = z1.as_array();
        let z2 = z2.as_array();
        let pos = pos.as_array();
        summator_fourier(spectrum_factor, modes, z1, z2, pos, num_threads).into_pyarray(py)
    }

    #[pyfunction(name = "calc_field_krige_and_variance")]
    fn calc_field_krige_and_variance_py<'py>(
        py: Python<'py>,
        krige_mat: PyReadonlyArray2<f64>,
        krig_vecs: PyReadonlyArray2<f64>,
        cond: PyReadonlyArray1<f64>,
        num_threads: Option<usize>,
    ) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
        let krige_mat = krige_mat.as_array();
        let krig_vecs = krig_vecs.as_array();
        let cond = cond.as_array();
        let (field, error) =
            calculator_field_krige_and_variance(krige_mat, krig_vecs, cond, num_threads);
        let field = field.into_pyarray(py);
        let error = error.into_pyarray(py);
        (field, error)
    }

    #[pyfunction(name = "calc_field_krige")]
    fn calc_field_krige_py<'py>(
        py: Python<'py>,
        krige_mat: PyReadonlyArray2<f64>,
        krig_vecs: PyReadonlyArray2<f64>,
        cond: PyReadonlyArray1<f64>,
        num_threads: Option<usize>,
    ) -> Bound<'py, PyArray1<f64>> {
        let krige_mat = krige_mat.as_array();
        let krig_vecs = krig_vecs.as_array();
        let cond = cond.as_array();
        calculator_field_krige(krige_mat, krig_vecs, cond, num_threads).into_pyarray(py)
    }

    #[pyfunction(name = "variogram_structured")]
    fn variogram_structured_py<'py>(
        py: Python<'py>,
        f: PyReadonlyArray2<f64>,
        estimator_type: Option<char>,
        num_threads: Option<usize>,
    ) -> Bound<'py, PyArray1<f64>> {
        let f = f.as_array();
        let estimator_type = estimator_type.unwrap_or('m');
        variogram_structured(f, estimator_type, num_threads).into_pyarray(py)
    }

    #[pyfunction(name = "variogram_ma_structured")]
    fn variogram_ma_structured_py<'py>(
        py: Python<'py>,
        f: PyReadonlyArray2<f64>,
        mask: PyReadonlyArray2<bool>,
        estimator_type: Option<char>,
        num_threads: Option<usize>,
    ) -> Bound<'py, PyArray1<f64>> {
        let f = f.as_array();
        let mask = mask.as_array();
        let estimator_type = estimator_type.unwrap_or('m');
        variogram_ma_structured(f, mask, estimator_type, num_threads).into_pyarray(py)
    }

    #[pyfunction(name = "variogram_directional")]
    #[allow(clippy::too_many_arguments)]
    fn variogram_directional_py<'py>(
        py: Python<'py>,
        f: PyReadonlyArray2<f64>,
        bin_edges: PyReadonlyArray1<f64>,
        pos: PyReadonlyArray2<f64>,
        direction: PyReadonlyArray2<f64>, //should be normed
        angles_tol: Option<f64>,
        bandwidth: Option<f64>,
        separate_dirs: Option<bool>,
        estimator_type: Option<char>,
        num_threads: Option<usize>,
    ) -> (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<u64>>) {
        let f = f.as_array();
        let bin_edges = bin_edges.as_array();
        let pos = pos.as_array();
        let direction = direction.as_array();
        let angles_tol = angles_tol.unwrap_or(std::f64::consts::PI / 8.0);
        let bandwidth = bandwidth.unwrap_or(-1.0);
        let separate_dirs = separate_dirs.unwrap_or(false);
        let estimator_type = estimator_type.unwrap_or('m');
        let (variogram, counts) = variogram_directional(
            f,
            bin_edges,
            pos,
            direction,
            angles_tol,
            bandwidth,
            separate_dirs,
            estimator_type,
            num_threads,
        );
        let variogram = variogram.into_pyarray(py);
        let counts = counts.into_pyarray(py);

        (variogram, counts)
    }

    #[pyfunction(name = "variogram_unstructured")]
    fn variogram_unstructured_py<'py>(
        py: Python<'py>,
        f: PyReadonlyArray2<f64>,
        bin_edges: PyReadonlyArray1<f64>,
        pos: PyReadonlyArray2<f64>,
        estimator_type: Option<char>,
        distance_type: Option<char>,
        num_threads: Option<usize>,
    ) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<u64>>) {
        let f = f.as_array();
        let bin_edges = bin_edges.as_array();
        let pos = pos.as_array();
        let estimator_type = estimator_type.unwrap_or('m');
        let distance_type = distance_type.unwrap_or('e');
        let (variogram, counts) = variogram_unstructured(
            f,
            bin_edges,
            pos,
            estimator_type,
            distance_type,
            num_threads,
        );
        let variogram = variogram.into_pyarray(py);
        let counts = counts.into_pyarray(py);

        (variogram, counts)
    }

    #[pyfunction(name = "mps_dist_block_cat")]
    #[allow(clippy::too_many_arguments)]
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

    #[pyfunction(name = "mps_dist_block_cat_rayon")]
    #[allow(clippy::too_many_arguments)]
    fn mps_dist_block_cat_rayon_py<'py>(
        py: Python<'py>,
        de_sim: PyReadonlyArray1<f64>,
        ti_flat: PyReadonlyArray1<f64>,
        base_flat: PyReadonlyArray1<i64>,
        lag_flat: PyReadonlyArray1<i64>,
        node_weights: PyReadonlyArray1<f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        dist_block_categorical_rayon(
            de_sim.as_array(),
            ti_flat.as_array(),
            base_flat.as_array(),
            lag_flat.as_array(),
            node_weights.as_array(),
        )
        .into_pyarray(py)
    }

    #[pyfunction(name = "mps_rayon_num_threads")]
    fn mps_rayon_num_threads_py() -> usize {
        rayon::current_num_threads()
    }

    #[pyfunction(name = "mps_dist_block_cat_masked")]
    #[allow(clippy::too_many_arguments)]
    fn mps_dist_block_cat_masked_py<'py>(
        py: Python<'py>,
        de_sim: PyReadonlyArray1<f64>,
        ti_flat: PyReadonlyArray1<f64>,
        base_flat: PyReadonlyArray1<i64>,
        lag_flat: PyReadonlyArray1<i64>,
        node_weights: PyReadonlyArray1<f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        dist_block_categorical_masked(
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

    #[pyfunction(name = "mps_dist_block_l1_masked")]
    #[allow(clippy::too_many_arguments)]
    fn mps_dist_block_l1_masked_py<'py>(
        py: Python<'py>,
        de_sim: PyReadonlyArray1<f64>,
        ti_flat: PyReadonlyArray1<f64>,
        base_flat: PyReadonlyArray1<i64>,
        lag_flat: PyReadonlyArray1<i64>,
        node_weights: PyReadonlyArray1<f64>,
        d_max: f64,
    ) -> Bound<'py, PyArray1<f64>> {
        dist_block_l1_masked(
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

    #[pyfunction(name = "mps_dist_block_l2_masked")]
    #[allow(clippy::too_many_arguments)]
    fn mps_dist_block_l2_masked_py<'py>(
        py: Python<'py>,
        de_sim: PyReadonlyArray1<f64>,
        ti_flat: PyReadonlyArray1<f64>,
        base_flat: PyReadonlyArray1<i64>,
        lag_flat: PyReadonlyArray1<i64>,
        node_weights: PyReadonlyArray1<f64>,
        d_max: f64,
    ) -> Bound<'py, PyArray1<f64>> {
        dist_block_l2_masked(
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

    #[pyfunction(name = "mps_dist_block_lp_masked")]
    #[allow(clippy::too_many_arguments)]
    fn mps_dist_block_lp_masked_py<'py>(
        py: Python<'py>,
        de_sim: PyReadonlyArray1<f64>,
        ti_flat: PyReadonlyArray1<f64>,
        base_flat: PyReadonlyArray1<i64>,
        lag_flat: PyReadonlyArray1<i64>,
        node_weights: PyReadonlyArray1<f64>,
        d_max: f64,
        p: f64,
    ) -> Bound<'py, PyArray1<f64>> {
        dist_block_lp_masked(
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

    #[pyfunction(name = "mps_dist_block_variation")]
    #[allow(clippy::too_many_arguments)]
    fn mps_dist_block_variation_py<'py>(
        py: Python<'py>,
        de_sim: PyReadonlyArray1<f64>,
        ti_flat: PyReadonlyArray1<f64>,
        base_flat: PyReadonlyArray1<i64>,
        lag_flat: PyReadonlyArray1<i64>,
        node_weights: PyReadonlyArray1<f64>,
        d_max: f64,
        p: f64,
    ) -> Bound<'py, PyArray1<f64>> {
        dist_block_variation(
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

    #[pyfunction(name = "mps_dist_block_variation_masked")]
    #[allow(clippy::too_many_arguments)]
    fn mps_dist_block_variation_masked_py<'py>(
        py: Python<'py>,
        de_sim: PyReadonlyArray1<f64>,
        ti_flat: PyReadonlyArray1<f64>,
        base_flat: PyReadonlyArray1<i64>,
        lag_flat: PyReadonlyArray1<i64>,
        node_weights: PyReadonlyArray1<f64>,
        d_max: f64,
        p: f64,
    ) -> Bound<'py, PyArray1<f64>> {
        dist_block_variation_masked(
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

    #[pyfunction(name = "mps_scan_node")]
    #[allow(clippy::too_many_arguments)]
    fn mps_scan_node_py<'py>(
        py: Python<'py>,
        lo: PyReadonlyArray1<i64>,
        win_shape: PyReadonlyArray1<i64>,
        start: usize,
        max_scan: usize,
        threshold: f64,
        ti_flat: PyReadonlyArray2<f64>,
        ti_strides: PyReadonlyArray1<i64>,
        active_ti_rows: PyReadonlyArray1<i64>,
        event_offsets: PyReadonlyArray1<i64>,
        de_sim: PyReadonlyArray1<f64>,
        lag_flat: PyReadonlyArray1<i64>,
        node_weights: PyReadonlyArray1<f64>,
        variable_weights: PyReadonlyArray1<f64>,
        metric_kinds: PyReadonlyArray1<i64>,
        has_nan: PyReadonlyArray1<u8>,
        d_max: PyReadonlyArray1<f64>,
        p_norm: PyReadonlyArray1<f64>,
        target_ti_rows: PyReadonlyArray1<i64>,
        check_centers: bool,
    ) -> Option<Bound<'py, PyArray1<i64>>> {
        scan_node(
            lo.as_array(),
            win_shape.as_array(),
            start,
            max_scan,
            threshold,
            ti_flat.as_array(),
            ti_strides.as_array(),
            active_ti_rows.as_array(),
            event_offsets.as_array(),
            de_sim.as_array(),
            lag_flat.as_array(),
            node_weights.as_array(),
            variable_weights.as_array(),
            metric_kinds.as_array(),
            has_nan.as_array(),
            d_max.as_array(),
            p_norm.as_array(),
            target_ti_rows.as_array(),
            check_centers,
        )
        .map(|arr| arr.into_pyarray(py))
    }

    #[pyfunction(name = "mps_simulate")]
    #[allow(clippy::too_many_arguments)]
    fn mps_simulate_py<'py>(
        py: Python<'py>,
        ti: PyReadonlyArray2<f64>,
        ti_shape: PyReadonlyArray1<i64>,
        initial_fields: PyReadonlyArray2<f64>,
        conditioned: PyReadonlyArray2<u8>,
        sim_shape: PyReadonlyArray1<i64>,
        path: PyReadonlyArray2<i64>,
        lag_matrices: PyReadonlyArray3<f64>,
        u_start: PyReadonlyArray1<f64>,
        u_fallback: PyReadonlyArray2<f64>,
        offsets: PyReadonlyArray2<i64>,
        n_neighbors: PyReadonlyArray1<i64>,
        max_radius: PyReadonlyArray1<f64>,
        variable_weights: PyReadonlyArray1<f64>,
        metric_kinds: PyReadonlyArray1<i64>,
        has_nan: PyReadonlyArray1<u8>,
        d_max: PyReadonlyArray1<f64>,
        p_norm: PyReadonlyArray1<f64>,
        threshold: f64,
        scan_fraction: f64,
        distance_power: f64,
        cond_weight: f64,
        partial_boundary: bool,
        num_threads: usize,
    ) -> PyResult<MpsSimulationPyOutput<'py>> {
        // Own the arrays before detaching so Python code may run concurrently
        // without being able to mutate buffers borrowed by the Rust engine.
        // This deliberately copies the full TI and all run inputs once per
        // simulation. It is the GIL-safety boundary, not an allocation-free
        // path; release memory claims must include these Rust-owned copies.
        let ti = ti.as_array().to_owned();
        let ti_shape = ti_shape.as_array().to_owned();
        let initial_fields = initial_fields.as_array().to_owned();
        let conditioned = conditioned.as_array().to_owned();
        let sim_shape = sim_shape.as_array().to_owned();
        let path = path.as_array().to_owned();
        let lag_matrices = lag_matrices.as_array().to_owned();
        let u_start = u_start.as_array().to_owned();
        let u_fallback = u_fallback.as_array().to_owned();
        let offsets = offsets.as_array().to_owned();
        let n_neighbors = n_neighbors.as_array().to_owned();
        let max_radius = max_radius.as_array().to_owned();
        let variable_weights = variable_weights.as_array().to_owned();
        let metric_kinds = metric_kinds.as_array().to_owned();
        let has_nan = has_nan.as_array().to_owned();
        let d_max = d_max.as_array().to_owned();
        let p_norm = p_norm.as_array().to_owned();

        let output = py
            .detach(move || {
                simulate_engine(
                    ti.view(),
                    ti_shape.view(),
                    initial_fields.view(),
                    conditioned.view(),
                    sim_shape.view(),
                    path.view(),
                    lag_matrices.view(),
                    u_start.view(),
                    u_fallback.view(),
                    offsets.view(),
                    n_neighbors.view(),
                    max_radius.view(),
                    variable_weights.view(),
                    metric_kinds.view(),
                    has_nan.view(),
                    d_max.view(),
                    p_norm.view(),
                    threshold,
                    scan_fraction,
                    distance_power,
                    cond_weight,
                    partial_boundary,
                    num_threads,
                )
            })
            .map_err(PyValueError::new_err)?;
        Ok((
            output.fields.into_pyarray(py),
            output.strict_fallback_count,
            output.level_count,
            output.max_ready_width,
            output.used_threads,
            output.collapsed_lag_count,
        ))
    }
}
