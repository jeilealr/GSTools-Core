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
mod short_vec;
pub mod variogram;

#[pymodule]
mod gstools_core {
    use crate::field::{summator, summator_fourier, summator_incompr};
    use crate::krige::{calculator_field_krige, calculator_field_krige_and_variance};
    use crate::mps::{
        dist_block_categorical, dist_block_l1, dist_block_l2, dist_block_lp, scan_node_categorical,
    };
    use crate::variogram::{
        variogram_directional, variogram_ma_structured, variogram_structured,
        variogram_unstructured,
    };
    use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
    use pyo3::prelude::*;

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
}
