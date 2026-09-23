"""Smoke-test the MPS exports of an installed gstools-core wheel."""

from __future__ import annotations

import numpy as np

import gstools_core


REQUIRED_EXPORTS = (
    "mps_dist_block_cat",
    "mps_dist_block_cat_masked",
    "mps_dist_block_l1",
    "mps_dist_block_l1_masked",
    "mps_dist_block_l2",
    "mps_dist_block_l2_masked",
    "mps_dist_block_lp",
    "mps_dist_block_lp_masked",
    "mps_dist_block_variation",
    "mps_dist_block_variation_masked",
    "mps_scan_node",
    "mps_scan_node_cat",
    "mps_simulate",
)


def main() -> None:
    missing = [name for name in REQUIRED_EXPORTS if not hasattr(gstools_core, name)]
    if missing:
        raise RuntimeError(f"installed wheel is missing MPS exports: {missing}")

    distance = gstools_core.mps_dist_block_cat(
        np.array([0.1]),
        np.array([0.2]),
        np.array([0], dtype=np.int64),
        np.array([0], dtype=np.int64),
        np.array([1.0]),
    )
    np.testing.assert_array_equal(distance, np.array([1.0]))

    result = gstools_core.mps_simulate(
        np.array([[0.0, 1.0]]),
        np.array([2], dtype=np.int64),
        np.array([[np.nan]]),
        np.zeros((1, 1), dtype=np.uint8),
        np.array([1], dtype=np.int64),
        np.array([[0]], dtype=np.int64),
        np.empty((0, 1, 1)),
        np.array([0.0]),
        np.array([[0.1]]),
        np.empty((0, 1), dtype=np.int64),
        np.array([1], dtype=np.int64),
        np.array([np.nan]),
        np.array([1.0]),
        np.array([0], dtype=np.int64),
        np.array([0], dtype=np.uint8),
        np.array([1.0]),
        np.array([1.0]),
        0.0,
        1.0,
        0.0,
        1.0,
        False,
        1,
    )
    field, strict_fallbacks, levels, ready_width, threads, collapsed = result
    np.testing.assert_array_equal(field, np.array([[0.0]]))
    assert (strict_fallbacks, levels, ready_width, threads, collapsed) == (
        0,
        1,
        1,
        1,
        0,
    )
    print("gstools-core MPS wheel smoke test: PASS")


if __name__ == "__main__":
    main()
