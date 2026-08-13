//! Rust-owned Direct Sampling simulation engine.
//!
//! Python intentionally keeps path and random-number generation during the
//! migration. This module owns the per-node computation and a
//! deterministic level scheduler: neighbour selection, search-window
//! construction, candidate scanning, retrieval, variation adjustment, and SG
//! writes. Nodes in one dependency level are independent and may be evaluated
//! by one simulation-local Rayon pool; their results are committed in path
//! order after the level completes.

use crate::mps::scan_node;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayView3};
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use rayon::{ThreadPool, ThreadPoolBuilder};

/// Result and scheduler diagnostics from one Rust-owned simulation.
pub struct SimulationOutput {
    /// Simulated variables as C-order rows, shape `(n_variables, sg_size)`.
    pub fields: Array2<f64>,
    /// Number of strict windows that required the established partial fallback.
    pub strict_fallback_count: usize,
    /// Number of deterministic dependency levels executed.
    pub level_count: usize,
    /// Largest number of mutually independent nodes in one level.
    pub max_ready_width: usize,
    /// Effective Rayon worker count after the measured small-path cutoff.
    pub used_threads: usize,
    /// Transformed event lags removed because multiple SG lags collapsed.
    pub collapsed_lag_count: usize,
}

#[derive(Clone)]
struct Neighbor {
    flat: usize,
    lag: Vec<i64>,
}

struct CachedNode {
    by_variable: Vec<Vec<Neighbor>>,
    level: usize,
}

struct Event {
    values: Vec<f64>,
    conditioned: Vec<bool>,
    lags: Vec<Vec<i64>>,
    lag_norms: Vec<f64>,
}

impl Event {
    fn len(&self) -> usize {
        self.values.len()
    }

    fn truncate(&mut self, keep: usize) {
        self.values.truncate(keep);
        self.conditioned.truncate(keep);
        self.lags.truncate(keep);
        self.lag_norms.truncate(keep);
    }
}

struct NodeResult {
    flat: usize,
    values: Vec<(usize, f64)>,
    strict_fallback_count: usize,
    collapsed_lag_count: usize,
}

type WindowBounds = Option<(Vec<i64>, Vec<i64>, usize)>;

fn c_strides(shape: ArrayView1<'_, i64>) -> Vec<i64> {
    let mut strides = vec![1_i64; shape.len()];
    for axis in (0..shape.len().saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1] * shape[axis + 1];
    }
    strides
}

fn flat_index(coords: &[i64], strides: &[i64]) -> usize {
    flat_index_iter(coords.iter().copied(), strides)
}

fn flat_index_iter(coords: impl Iterator<Item = i64>, strides: &[i64]) -> usize {
    coords
        .zip(strides.iter())
        .map(|(coordinate, &stride)| coordinate * stride)
        .sum::<i64>() as usize
}

#[allow(clippy::too_many_arguments)]
fn build_neighbor_cache(
    path: ArrayView2<'_, i64>,
    sim_shape: ArrayView1<'_, i64>,
    offsets: ArrayView2<'_, i64>,
    conditioned: ArrayView2<'_, u8>,
    n_neighbors: ArrayView1<'_, i64>,
    max_radius: ArrayView1<'_, f64>,
) -> Vec<CachedNode> {
    let variable_count = conditioned.nrows();
    let sg_size = sim_shape.iter().product::<i64>() as usize;
    let strides = c_strides(sim_shape);
    let path_flat: Vec<usize> = path
        .rows()
        .into_iter()
        .map(|coords| flat_index_iter(coords.iter().copied(), &strides))
        .collect();

    let mut path_maps = vec![vec![-1_i64; sg_size]; variable_count];
    for variable in 0..variable_count {
        for (index, &flat) in path_flat.iter().enumerate() {
            path_maps[variable][flat] = index as i64;
        }
        for flat in 0..sg_size {
            if conditioned[(variable, flat)] != 0 {
                path_maps[variable][flat] = -1;
            }
        }
    }

    let mut levels = vec![0_usize; path.nrows()];
    let mut cache = Vec::with_capacity(path.nrows());
    for node in 0..path.nrows() {
        let coords = path.row(node);
        let mut by_variable = Vec::with_capacity(variable_count);
        let mut level = 0_usize;
        for variable in 0..variable_count {
            let radius_sq = if max_radius[variable].is_finite() {
                Some(max_radius[variable] * max_radius[variable])
            } else {
                None
            };
            let mut selected = Vec::with_capacity(n_neighbors[variable] as usize);
            for offset in offsets.rows() {
                let distance_sq: f64 = offset
                    .iter()
                    .map(|&component| (component * component) as f64)
                    .sum();
                if radius_sq.is_some_and(|limit| distance_sq > limit) {
                    break;
                }
                let mut candidate = Vec::with_capacity(sim_shape.len());
                let mut in_bounds = true;
                for axis in 0..sim_shape.len() {
                    let coordinate = coords[axis] + offset[axis];
                    if coordinate < 0 || coordinate >= sim_shape[axis] {
                        in_bounds = false;
                        break;
                    }
                    candidate.push(coordinate);
                }
                if !in_bounds {
                    continue;
                }
                let flat = flat_index(&candidate, &strides);
                let dependency = path_maps[variable][flat];
                if dependency == -1 || dependency < node as i64 {
                    if dependency >= 0 {
                        level = level.max(levels[dependency as usize] + 1);
                    }
                    selected.push(Neighbor {
                        flat,
                        lag: offset.to_vec(),
                    });
                    if selected.len() == n_neighbors[variable] as usize {
                        break;
                    }
                }
            }
            by_variable.push(selected);
        }
        levels[node] = level;
        cache.push(CachedNode { by_variable, level });
    }
    cache
}

fn compute_node_weights(
    lag_norms: &[f64],
    conditioned: &[bool],
    distance_power: f64,
    cond_weight: f64,
) -> Vec<f64> {
    let mut weights = vec![1.0_f64; lag_norms.len()];
    if distance_power != 0.0 {
        for (weight, &norm) in weights.iter_mut().zip(lag_norms.iter()) {
            if norm != 0.0 {
                *weight = norm.powf(-distance_power);
            }
        }
    }
    for (weight, &is_conditioned) in weights.iter_mut().zip(conditioned.iter()) {
        if is_conditioned {
            *weight *= cond_weight;
        }
    }
    let total: f64 = weights.iter().sum();
    if total == 0.0 {
        weights.fill(0.0);
    } else if !total.is_finite() {
        let uniform = 1.0 / weights.len() as f64;
        weights.fill(uniform);
    } else {
        for weight in &mut weights {
            *weight /= total;
        }
    }
    weights
}

fn bounds_for_prefix(
    lags: &[Vec<i64>],
    keep: usize,
    ti_shape: ArrayView1<'_, i64>,
) -> Option<(Vec<i64>, Vec<i64>)> {
    let dim = ti_shape.len();
    let nonzero: Vec<&Vec<i64>> = lags[..keep]
        .iter()
        .filter(|lag| lag.iter().any(|&component| component != 0))
        .collect();
    if nonzero.is_empty() {
        return Some((
            vec![0_i64; dim],
            ti_shape.iter().map(|&size| size - 1).collect(),
        ));
    }
    let mut lo = vec![0_i64; dim];
    let mut hi: Vec<i64> = ti_shape.iter().map(|&size| size - 1).collect();
    for axis in 0..dim {
        let min_lag = nonzero.iter().map(|lag| lag[axis]).min().unwrap_or(0);
        let max_lag = nonzero.iter().map(|lag| lag[axis]).max().unwrap_or(0);
        lo[axis] = 0_i64.max(-min_lag);
        hi[axis] = (ti_shape[axis] - 1).min(ti_shape[axis] - 1 - max_lag);
    }
    lo.iter()
        .zip(hi.iter())
        .all(|(&lower, &upper)| lower <= upper)
        .then_some((lo, hi))
}

fn window_bounds(
    lags: &[Vec<i64>],
    ti_shape: ArrayView1<'_, i64>,
    partial_boundary: bool,
) -> (WindowBounds, bool) {
    if let Some((lo, hi)) = bounds_for_prefix(lags, lags.len(), ti_shape) {
        return (Some((lo, hi, lags.len())), false);
    }
    let strict_fallback = !partial_boundary;
    for keep in (1..lags.len()).rev() {
        if let Some((lo, hi)) = bounds_for_prefix(lags, keep, ti_shape) {
            return (Some((lo, hi, keep)), strict_fallback);
        }
    }
    (None, strict_fallback)
}

fn transform_and_reduce_event(
    event: &mut Event,
    matrix: ArrayView2<'_, f64>,
    ti_shape: ArrayView1<'_, i64>,
) -> usize {
    let mut transformed = Vec::with_capacity(event.len());
    for lag in &event.lags {
        let mut mapped = vec![0_i64; lag.len()];
        for output_axis in 0..lag.len() {
            let value: f64 = (0..lag.len())
                .map(|input_axis| lag[input_axis] as f64 * matrix[(output_axis, input_axis)])
                .sum();
            mapped[output_axis] = value.round_ties_even() as i64;
        }
        transformed.push(mapped);
    }

    let mut keep = Vec::with_capacity(transformed.len());
    for index in 0..transformed.len() {
        if !keep
            .iter()
            .any(|&prior| transformed[prior] == transformed[index])
        {
            keep.push(index);
        }
    }
    let collapsed = transformed.len() - keep.len();
    if collapsed > 0 {
        event.values = keep.iter().map(|&index| event.values[index]).collect();
        event.conditioned = keep.iter().map(|&index| event.conditioned[index]).collect();
        event.lag_norms = keep.iter().map(|&index| event.lag_norms[index]).collect();
        transformed = keep
            .iter()
            .map(|&index| transformed[index].clone())
            .collect();
    }
    event.lags = transformed;

    loop {
        let nonzero: Vec<usize> = event
            .lags
            .iter()
            .enumerate()
            .filter_map(|(index, lag)| lag.iter().any(|&component| component != 0).then_some(index))
            .collect();
        if nonzero.is_empty() {
            break;
        }
        let mut over = vec![0_i64; ti_shape.len()];
        for axis in 0..ti_shape.len() {
            let min_lag = nonzero
                .iter()
                .map(|&index| event.lags[index][axis])
                .min()
                .unwrap_or(0)
                .min(0);
            let max_lag = nonzero
                .iter()
                .map(|&index| event.lags[index][axis])
                .max()
                .unwrap_or(0)
                .max(0);
            over[axis] = max_lag - min_lag - (ti_shape[axis] - 1);
        }
        if over.iter().all(|&amount| amount <= 0) {
            break;
        }
        let axis = over
            .iter()
            .enumerate()
            .max_by_key(|&(index, &amount)| (amount, std::cmp::Reverse(index)))
            .map(|(index, _)| index)
            .unwrap_or(0);
        let drop = nonzero
            .iter()
            .copied()
            .max_by_key(|&index| (event.lags[index][axis].abs(), std::cmp::Reverse(index)))
            .unwrap_or(0);
        event.values.remove(drop);
        event.conditioned.remove(drop);
        event.lags.remove(drop);
        event.lag_norms.remove(drop);
    }
    collapsed
}

fn nanmean(values: impl Iterator<Item = f64>) -> Option<f64> {
    let mut sum = 0.0_f64;
    let mut count = 0_usize;
    for value in values {
        if !value.is_nan() {
            sum += value;
            count += 1;
        }
    }
    (count > 0).then_some(sum / count as f64)
}

fn fallback_flat(
    node: usize,
    u_fallback: ArrayView2<'_, f64>,
    ti_shape: ArrayView1<'_, i64>,
    ti_strides: &[i64],
    finite_all: &[usize],
    ti_has_nan: bool,
) -> usize {
    if ti_has_nan {
        let index = (u_fallback[(node, 0)] * finite_all.len() as f64) as usize;
        finite_all[index]
    } else {
        let coords: Vec<i64> = (0..ti_shape.len())
            .map(|axis| (u_fallback[(node, axis)] * ti_shape[axis] as f64) as i64)
            .collect();
        flat_index(&coords, ti_strides)
    }
}

#[allow(clippy::too_many_arguments)]
fn fallback_result(
    node: usize,
    node_flat: usize,
    targets: &[usize],
    ti: ArrayView2<'_, f64>,
    u_fallback: ArrayView2<'_, f64>,
    ti_shape: ArrayView1<'_, i64>,
    ti_strides: &[i64],
    finite_all: &[usize],
    ti_has_nan: bool,
    strict_fallback_count: usize,
) -> NodeResult {
    let source = fallback_flat(
        node, u_fallback, ti_shape, ti_strides, finite_all, ti_has_nan,
    );
    NodeResult {
        flat: node_flat,
        values: targets
            .iter()
            .map(|&variable| (variable, ti[(variable, source)]))
            .collect(),
        strict_fallback_count,
        collapsed_lag_count: 0,
    }
}

#[allow(clippy::too_many_arguments)]
fn simulate_node(
    node: usize,
    cached: &CachedNode,
    fields: ArrayView2<'_, f64>,
    ti: ArrayView2<'_, f64>,
    ti_shape: ArrayView1<'_, i64>,
    sim_shape: ArrayView1<'_, i64>,
    path: ArrayView2<'_, i64>,
    lag_matrices: ArrayView3<'_, f64>,
    u_start: ArrayView1<'_, f64>,
    u_fallback: ArrayView2<'_, f64>,
    conditioned: ArrayView2<'_, u8>,
    variable_weights: ArrayView1<'_, f64>,
    metric_kinds: ArrayView1<'_, i64>,
    has_nan: ArrayView1<'_, u8>,
    d_max: ArrayView1<'_, f64>,
    p_norm: ArrayView1<'_, f64>,
    threshold: f64,
    scan_fraction: f64,
    distance_power: f64,
    cond_weight: f64,
    partial_boundary: bool,
    finite_all: &[usize],
    ti_has_nan: bool,
) -> NodeResult {
    let ti_strides = c_strides(ti_shape);
    let sim_strides = c_strides(sim_shape);
    let node_coords = path.row(node);
    let node_flat = flat_index_iter(node_coords.iter().copied(), &sim_strides);
    let targets: Vec<usize> = (0..fields.nrows())
        .filter(|&variable| fields[(variable, node_flat)].is_nan())
        .collect();

    let mut events = Vec::with_capacity(fields.nrows());
    let mut collapsed_lag_count = 0_usize;
    for variable in 0..fields.nrows() {
        let collocated = !fields[(variable, node_flat)].is_nan();
        let capacity = cached.by_variable[variable].len() + usize::from(collocated);
        let mut event = Event {
            values: Vec::with_capacity(capacity),
            conditioned: Vec::with_capacity(capacity),
            lags: Vec::with_capacity(capacity),
            lag_norms: Vec::with_capacity(capacity),
        };
        if collocated {
            event.values.push(fields[(variable, node_flat)]);
            event
                .conditioned
                .push(conditioned[(variable, node_flat)] != 0);
            event.lags.push(vec![0_i64; sim_shape.len()]);
            event.lag_norms.push(0.0);
        }
        for neighbor in &cached.by_variable[variable] {
            event.values.push(fields[(variable, neighbor.flat)]);
            event
                .conditioned
                .push(conditioned[(variable, neighbor.flat)] != 0);
            event.lags.push(neighbor.lag.clone());
            event.lag_norms.push(
                neighbor
                    .lag
                    .iter()
                    .map(|&component| (component * component) as f64)
                    .sum::<f64>()
                    .sqrt(),
            );
        }
        if !lag_matrices.is_empty() {
            collapsed_lag_count += transform_and_reduce_event(
                &mut event,
                lag_matrices.slice(ndarray::s![node, .., ..]),
                ti_shape,
            );
        }
        events.push(event);
    }

    if events.iter().all(|event| event.len() == 0) {
        let mut result = fallback_result(
            node,
            node_flat,
            &targets,
            ti,
            u_fallback,
            ti_shape,
            &ti_strides,
            finite_all,
            ti_has_nan,
            0,
        );
        result.collapsed_lag_count = collapsed_lag_count;
        return result;
    }

    let mut win_lo = vec![0_i64; ti_shape.len()];
    let mut win_hi: Vec<i64> = ti_shape.iter().map(|&size| size - 1).collect();
    let mut strict_fallback_count = 0_usize;
    for event in &mut events {
        let (bounds, strict_fallback) = window_bounds(&event.lags, ti_shape, partial_boundary);
        if strict_fallback {
            strict_fallback_count += 1;
        }
        let Some((lo, hi, keep)) = bounds else {
            let mut result = fallback_result(
                node,
                node_flat,
                &targets,
                ti,
                u_fallback,
                ti_shape,
                &ti_strides,
                finite_all,
                ti_has_nan,
                strict_fallback_count,
            );
            result.collapsed_lag_count = collapsed_lag_count;
            return result;
        };
        event.truncate(keep);
        for axis in 0..ti_shape.len() {
            win_lo[axis] = win_lo[axis].max(lo[axis]);
            win_hi[axis] = win_hi[axis].min(hi[axis]);
        }
    }
    if win_lo.iter().zip(win_hi.iter()).any(|(&lo, &hi)| lo > hi) {
        let mut result = fallback_result(
            node,
            node_flat,
            &targets,
            ti,
            u_fallback,
            ti_shape,
            &ti_strides,
            finite_all,
            ti_has_nan,
            strict_fallback_count,
        );
        result.collapsed_lag_count = collapsed_lag_count;
        return result;
    }

    let active: Vec<usize> = (0..events.len())
        .filter(|&variable| events[variable].len() > 0)
        .collect();
    let mut event_offsets = Vec::with_capacity(active.len() + 1);
    let mut de_sim = Vec::new();
    let mut lag_flat = Vec::new();
    let mut node_weights = Vec::new();
    event_offsets.push(0_i64);
    for &variable in &active {
        let event = &events[variable];
        de_sim.extend_from_slice(&event.values);
        lag_flat.extend(event.lags.iter().map(|lag| {
            lag.iter()
                .zip(ti_strides.iter())
                .map(|(&component, &stride)| component * stride)
                .sum::<i64>()
        }));
        node_weights.extend(compute_node_weights(
            &event.lag_norms,
            &event.conditioned,
            distance_power,
            cond_weight,
        ));
        event_offsets.push(de_sim.len() as i64);
    }

    let win_shape: Vec<i64> = win_lo
        .iter()
        .zip(win_hi.iter())
        .map(|(&lo, &hi)| hi - lo + 1)
        .collect();
    let win_size = win_shape.iter().product::<i64>() as usize;
    let ti_size = ti_shape.iter().product::<i64>() as usize;
    let max_scan = 1_usize.max(win_size.min((scan_fraction * ti_size as f64) as usize));
    let start = (u_start[node] * win_size as f64) as usize;

    let active_rows = Array1::from_iter(active.iter().map(|&value| value as i64));
    let active_variable_weights =
        Array1::from_iter(active.iter().map(|&value| variable_weights[value]));
    let active_metrics = Array1::from_iter(active.iter().map(|&value| metric_kinds[value]));
    let active_has_nan = Array1::from_iter(active.iter().map(|&value| has_nan[value]));
    let active_d_max = Array1::from_iter(active.iter().map(|&value| d_max[value]));
    let active_p_norm = Array1::from_iter(active.iter().map(|&value| p_norm[value]));
    let target_rows = Array1::from_iter(targets.iter().map(|&value| value as i64));

    let anchor = scan_node(
        Array1::from_vec(win_lo).view(),
        Array1::from_vec(win_shape).view(),
        start,
        max_scan,
        threshold,
        ti,
        Array1::from_vec(ti_strides.clone()).view(),
        active_rows.view(),
        Array1::from_vec(event_offsets).view(),
        Array1::from_vec(de_sim).view(),
        Array1::from_vec(lag_flat).view(),
        Array1::from_vec(node_weights).view(),
        active_variable_weights.view(),
        active_metrics.view(),
        active_has_nan.view(),
        active_d_max.view(),
        active_p_norm.view(),
        target_rows.view(),
        ti_has_nan,
    );
    let Some(anchor) = anchor else {
        let mut result = fallback_result(
            node,
            node_flat,
            &targets,
            ti,
            u_fallback,
            ti_shape,
            &ti_strides,
            finite_all,
            ti_has_nan,
            strict_fallback_count,
        );
        result.collapsed_lag_count = collapsed_lag_count;
        return result;
    };
    let anchor_flat = flat_index(anchor.as_slice().expect("contiguous anchor"), &ti_strides);

    let values = targets
        .iter()
        .map(|&variable| {
            let center = ti[(variable, anchor_flat)];
            if metric_kinds[variable] != 2 || events[variable].len() == 0 {
                return (variable, center);
            }
            let ti_event = events[variable].lags.iter().map(|lag| {
                let lag_flat: i64 = lag
                    .iter()
                    .zip(ti_strides.iter())
                    .map(|(&component, &stride)| component * stride)
                    .sum();
                ti[(variable, (anchor_flat as i64 + lag_flat) as usize)]
            });
            let ti_values: Vec<f64> = ti_event.collect();
            let ti_mean = if ti_values.iter().any(|value| value.is_finite()) {
                nanmean(ti_values.iter().copied()).unwrap_or(0.0)
            } else {
                0.0
            };
            let sim_mean = nanmean(events[variable].values.iter().copied()).unwrap_or(0.0);
            (variable, center - ti_mean + sim_mean)
        })
        .collect();

    NodeResult {
        flat: node_flat,
        values,
        strict_fallback_count,
        collapsed_lag_count,
    }
}

/// Run one complete MPS simulation with a deterministic scheduler.
///
/// Python supplies the simulation path and per-node random values so seeded
/// behavior remains stable during migration. Rust owns all per-node work. The
/// path-derived neighbour graph is converted to fixed dependency levels;
/// nodes inside one level are independent, calculated in parallel, and then
/// committed in path order. Candidate/lag reductions remain sequential.
///
/// `max_radius` uses `NaN` to represent Python's `None`. `metric_kinds` uses
/// `0` for categorical, `1` for continuous Lp, and `2` for variation.
/// `partial_boundary == false` means strict mode with the established partial
/// fallback when the complete event cannot fit.
///
/// # Arguments
///
/// * `ti` – Float64 TI rows, shape `(n_variables, ti_size)`.
/// * `ti_shape` – Shared TI shape.
/// * `initial_fields` – Initial SG rows with unknown cells as `NaN`.
/// * `conditioned` – Conditioning mask, shape matching `initial_fields`.
/// * `sim_shape` – Shared SG shape.
/// * `path` – Pre-generated unknown-node path, shape `(n_nodes, dim)`.
/// * `lag_matrices` – Optional per-node SG-to-TI matrices, shape `(n_nodes, dim, dim)`;
///   an empty first axis selects the stationary identity path.
/// * `u_start` – Per-node scan-start uniforms.
/// * `u_fallback` – Per-node fallback uniforms, shape `(n_nodes, dim)`.
/// * `offsets` – Canonically ordered nonzero SG offsets.
/// * `n_neighbors` – Maximum event size per variable.
/// * `max_radius` – Radius per variable (`NaN` means unlimited).
/// * `variable_weights` – Normalized multivariate weights.
/// * `metric_kinds` – Distance kind per variable.
/// * `has_nan` – Per-variable masked-distance flags.
/// * `d_max` – Continuous normalization ranges.
/// * `p_norm` – Lp or variation exponent per variable.
/// * `threshold` – DSBC when nonpositive; strict DS threshold otherwise.
/// * `scan_fraction` – Fraction of full TI examined per node.
/// * `distance_power` – Spatial event-weight exponent.
/// * `cond_weight` – Conditioning-event weight multiplier.
/// * `partial_boundary` – Select partial instead of strict-first geometry.
/// * `num_threads` – Size of the one simulation-local Rayon pool.
///
/// # Returns
///
/// Simulated fields plus deterministic scheduler diagnostics, or an error for
/// inconsistent internal inputs or a generated `NaN` value.
#[allow(clippy::too_many_arguments)]
pub fn simulate_engine(
    ti: ArrayView2<'_, f64>,
    ti_shape: ArrayView1<'_, i64>,
    initial_fields: ArrayView2<'_, f64>,
    conditioned: ArrayView2<'_, u8>,
    sim_shape: ArrayView1<'_, i64>,
    path: ArrayView2<'_, i64>,
    lag_matrices: ArrayView3<'_, f64>,
    u_start: ArrayView1<'_, f64>,
    u_fallback: ArrayView2<'_, f64>,
    offsets: ArrayView2<'_, i64>,
    n_neighbors: ArrayView1<'_, i64>,
    max_radius: ArrayView1<'_, f64>,
    variable_weights: ArrayView1<'_, f64>,
    metric_kinds: ArrayView1<'_, i64>,
    has_nan: ArrayView1<'_, u8>,
    d_max: ArrayView1<'_, f64>,
    p_norm: ArrayView1<'_, f64>,
    threshold: f64,
    scan_fraction: f64,
    distance_power: f64,
    cond_weight: f64,
    partial_boundary: bool,
    num_threads: usize,
) -> Result<SimulationOutput, String> {
    let variable_count = ti.nrows();
    let ti_size = ti_shape.iter().product::<i64>() as usize;
    let sg_size = sim_shape.iter().product::<i64>() as usize;
    if variable_count == 0
        || ti.ncols() != ti_size
        || initial_fields.dim() != (variable_count, sg_size)
        || conditioned.dim() != initial_fields.dim()
        || path.ncols() != sim_shape.len()
        || (!lag_matrices.is_empty()
            && lag_matrices.dim() != (path.nrows(), sim_shape.len(), sim_shape.len()))
        || u_start.len() != path.nrows()
        || u_fallback.dim() != (path.nrows(), sim_shape.len())
        || offsets.ncols() != sim_shape.len()
        || n_neighbors.len() != variable_count
        || max_radius.len() != variable_count
        || variable_weights.len() != variable_count
        || metric_kinds.len() != variable_count
        || has_nan.len() != variable_count
        || d_max.len() != variable_count
        || p_norm.len() != variable_count
    {
        return Err("inconsistent MPS simulation-engine array shapes".to_owned());
    }

    let ti_has_nan = ti.iter().any(|value| value.is_nan());
    let finite_all: Vec<usize> = (0..ti_size)
        .filter(|&flat| (0..variable_count).all(|variable| !ti[(variable, flat)].is_nan()))
        .collect();
    if ti_has_nan && finite_all.is_empty() {
        return Err("training image has no jointly defined cell".to_owned());
    }

    let cache = build_neighbor_cache(
        path,
        sim_shape,
        offsets,
        conditioned,
        n_neighbors,
        max_radius,
    );
    let level_count = cache
        .iter()
        .map(|node| node.level)
        .max()
        .map_or(0, |level| level + 1);
    let mut levels = vec![Vec::new(); level_count];
    for (node, cached) in cache.iter().enumerate() {
        levels[cached.level].push(node);
    }
    let max_ready_width = levels.iter().map(Vec::len).max().unwrap_or(0);
    // Building/waking a pool regressed the measured 256-node small case. Keep
    // paths below 512 nodes serial; larger representative 2-D/3-D paths passed
    // the Action Plan 7 continuation gate.
    let used_threads = if num_threads > 1 && path.nrows() >= 512 {
        num_threads
    } else {
        1
    };
    let pool: Option<ThreadPool> = if used_threads > 1 {
        Some(
            ThreadPoolBuilder::new()
                .num_threads(used_threads)
                .build()
                .map_err(|error| format!("could not build MPS Rayon pool: {error}"))?,
        )
    } else {
        None
    };

    let mut fields = initial_fields.to_owned();
    let mut strict_fallback_count = 0_usize;
    let mut collapsed_lag_count = 0_usize;
    for ready in &levels {
        let calculate = |node: &usize| {
            simulate_node(
                *node,
                &cache[*node],
                fields.view(),
                ti,
                ti_shape,
                sim_shape,
                path,
                lag_matrices,
                u_start,
                u_fallback,
                conditioned,
                variable_weights,
                metric_kinds,
                has_nan,
                d_max,
                p_norm,
                threshold,
                scan_fraction,
                distance_power,
                cond_weight,
                partial_boundary,
                &finite_all,
                ti_has_nan,
            )
        };
        let results: Vec<NodeResult> = if let Some(pool) = &pool {
            pool.install(|| ready.par_iter().map(calculate).collect())
        } else {
            ready.iter().map(calculate).collect()
        };
        for result in results {
            strict_fallback_count += result.strict_fallback_count;
            collapsed_lag_count += result.collapsed_lag_count;
            for (variable, value) in result.values {
                if value.is_nan() {
                    return Err(format!(
                        "simulation produced NaN for variable {variable} at flat node {}",
                        result.flat
                    ));
                }
                fields[(variable, result.flat)] = value;
            }
        }
    }

    Ok(SimulationOutput {
        fields,
        strict_fallback_count,
        level_count,
        max_ready_width,
        used_threads,
        collapsed_lag_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array3};

    fn categorical_fixture(num_threads: usize) -> SimulationOutput {
        simulate_engine(
            array![[0.0, 1.0, 1.0, 0.0]].view(),
            array![4_i64].view(),
            array![[0.0, f64::NAN, f64::NAN]].view(),
            array![[1_u8, 0, 0]].view(),
            array![3_i64].view(),
            array![[1_i64], [2_i64]].view(),
            Array3::<f64>::zeros((0, 1, 1)).view(),
            array![0.0, 0.5].view(),
            array![[0.25], [0.75]].view(),
            array![[-1_i64], [1_i64]].view(),
            array![1_i64].view(),
            array![f64::NAN].view(),
            array![1.0].view(),
            array![0_i64].view(),
            array![0_u8].view(),
            array![1.0].view(),
            array![1.0].view(),
            0.0,
            1.0,
            0.0,
            1.0,
            false,
            num_threads,
        )
        .expect("valid fixture")
    }

    #[test]
    fn stationary_engine_preserves_conditioning_and_is_thread_independent() {
        let serial = categorical_fixture(1);
        let parallel = categorical_fixture(4);
        assert_eq!(serial.fields, parallel.fields);
        assert_eq!(serial.fields[(0, 0)], 0.0);
        assert!(serial.fields.iter().all(|value| value.is_finite()));
        assert_eq!(serial.level_count, parallel.level_count);
        assert_eq!(parallel.used_threads, 1);
    }

    #[test]
    fn transformed_event_keeps_first_duplicate_and_reduces_to_fit() {
        let mut event = Event {
            values: vec![5.0, 9.0, 12.0],
            conditioned: vec![true, false, false],
            lags: vec![vec![0, 1], vec![0, 2], vec![10, 0]],
            lag_norms: vec![1.0, 2.0, 10.0],
        };
        let collapsed = transform_and_reduce_event(
            &mut event,
            array![[1.0, 0.0], [0.0, 0.1]].view(),
            array![4_i64, 4_i64].view(),
        );
        assert_eq!(collapsed, 1);
        assert_eq!(event.lags, vec![vec![0, 0]]);
        assert_eq!(event.values, vec![5.0]);
        assert_eq!(event.conditioned, vec![true]);
    }

    #[test]
    fn strict_window_reports_fallback_even_when_no_prefix_fits() {
        let lags = vec![vec![10_i64]];
        let (bounds, strict_fallback) = window_bounds(&lags, array![4_i64].view(), false);
        assert!(bounds.is_none());
        assert!(strict_fallback);

        let (bounds, strict_fallback) = window_bounds(&lags, array![4_i64].view(), true);
        assert!(bounds.is_none());
        assert!(!strict_fallback);
    }

    #[test]
    fn variation_retrieval_applies_mean_shift() {
        let output = simulate_engine(
            array![[10.0, 11.0, 20.0, 21.0]].view(),
            array![4_i64].view(),
            array![[5.0, f64::NAN]].view(),
            array![[1_u8, 0]].view(),
            array![2_i64].view(),
            array![[1_i64]].view(),
            Array3::<f64>::zeros((0, 1, 1)).view(),
            array![0.0].view(),
            array![[0.0]].view(),
            array![[-1_i64], [1_i64]].view(),
            array![1_i64].view(),
            array![f64::NAN].view(),
            array![1.0].view(),
            array![2_i64].view(),
            array![0_u8].view(),
            array![11.0].view(),
            array![2.0].view(),
            1.0,
            1.0,
            0.0,
            1.0,
            false,
            1,
        )
        .expect("valid variation fixture");
        assert_eq!(output.fields[(0, 1)], 6.0);
    }
}
