"""Tests for batched Wasserstein distance bindings."""

import numpy as np

from ks_shadowing import pha
from ks_shadowing.core import INTEGRATION_DT
from ks_shadowing.core.rpo import RPO
from ks_shadowing.core.trajectory import KSTrajectory
from ks_shadowing.pha.detection import _compute_order_scales
from ks_shadowing.pha.persistence import KSPersistenceTrajectory
from ks_shadowing.pha.wasserstein import _wasserstein_column, wasserstein_matrix


def _flatten(diagrams: list) -> tuple:
    """Flatten ``diagrams`` via ``KSPersistenceTrajectory._flatten``."""
    return KSPersistenceTrajectory(diagrams=diagrams, dt=0.02)._flatten()


def test_flatten_offsets_for_variable_size_diagrams() -> None:
    """``_flatten`` produces cumulative ``offsets`` that index each diagram's
    starting row in the concatenated array."""
    diagrams = [
        np.array([[0.0, 1.0], [0.5, 2.0]], dtype=np.float64),
        np.zeros((0, 2), dtype=np.float64),
        np.array([[1.0, 3.0]], dtype=np.float64),
    ]
    flat, offsets = _flatten(diagrams)
    np.testing.assert_array_equal(offsets, [0, 2, 2, 3])
    assert flat.shape == (3, 2)
    np.testing.assert_array_equal(
        flat,
        np.array([[0.0, 1.0], [0.5, 2.0], [1.0, 3.0]], dtype=np.float64),
    )


def test_wasserstein_column_self_zero(rng: np.random.Generator) -> None:
    """``_wasserstein_column`` returns 0 for a diagram compared against
    itself."""
    modes = np.zeros((4, 17), dtype=np.complex128)
    modes[:, 1:16] = (rng.standard_normal((4, 15)) + 1j * rng.standard_normal((4, 15))) * 0.1
    trajectory = KSTrajectory(modes=modes, dt=0.02, resolution=32)
    diagrams = KSPersistenceTrajectory.from_trajectory(trajectory)
    flat, offsets = diagrams._flatten()

    column = _wasserstein_column(flat, offsets, diagrams.diagrams[0])
    assert column.shape == (4,)
    assert column[0] == 0.0


def test_wasserstein_column_empty_inputs() -> None:
    """``_wasserstein_column`` returns shape ``(0,)`` for an empty trajectory
    list, and accepts an empty ``diagram_b``."""
    flat_empty, offsets_empty = _flatten([])
    nonempty = np.array([[0.0, 1.0]], dtype=np.float64)
    assert _wasserstein_column(flat_empty, offsets_empty, nonempty).shape == (0,)

    flat_mix, offsets_mix = _flatten([np.zeros((0, 2)), np.array([[0.0, 1.0]], dtype=np.float64)])
    column = _wasserstein_column(flat_mix, offsets_mix, np.zeros((0, 2), dtype=np.float64))
    assert column.shape == (2,)


def test_wasserstein_matrix_is_transpose_symmetric(rng: np.random.Generator) -> None:
    """``wasserstein_matrix(a, b)`` equals ``wasserstein_matrix(b, a)``
    transposed, and comparing a set against itself gives a zero diagonal."""
    modes = np.zeros((7, 17), dtype=np.complex128)
    modes[:, 1:16] = (rng.standard_normal((7, 15)) + 1j * rng.standard_normal((7, 15))) * 0.1
    trajectory = KSTrajectory(modes=modes, dt=0.02, resolution=32)
    diagrams = KSPersistenceTrajectory.from_trajectory(trajectory)
    first = KSPersistenceTrajectory(diagrams=diagrams.diagrams[:3], dt=diagrams.dt)
    second = KSPersistenceTrajectory(diagrams=diagrams.diagrams[3:], dt=diagrams.dt)

    matrix = wasserstein_matrix(first, second)
    assert matrix.shape == (3, 4)
    np.testing.assert_allclose(matrix, wasserstein_matrix(second, first).T, rtol=1e-6, atol=1e-6)

    self_matrix = wasserstein_matrix(diagrams, diagrams)
    np.testing.assert_allclose(np.diag(self_matrix), 0.0, atol=1e-6)


def test_order_average_before_min_matches_compute_min_distances(
    small_rpos: list[RPO],
    sample_initial_state: np.ndarray,
) -> None:
    """Averaging per-order Wasserstein matrices before the minimum over phases
    and RPOs reproduces ``compute_min_distances`` at the same
    ``max_derivative_order``."""
    max_derivative_order = 1
    downsample = 20
    trajectory = KSTrajectory.from_initial_state(
        sample_initial_state, dt=INTEGRATION_DT, num_timesteps=60, resolution=16
    )

    trajectory_diagrams = [
        KSPersistenceTrajectory.from_trajectory(trajectory, order=order)
        for order in range(max_derivative_order + 1)
    ]

    reduced = np.full(trajectory.num_timesteps, np.inf)
    for rpo in small_rpos:
        rpo_trajectory = KSTrajectory.from_rpo(rpo, 16, downsample=downsample)
        stacked = np.stack(
            [
                wasserstein_matrix(
                    trajectory_diagrams[order],
                    KSPersistenceTrajectory.from_trajectory(rpo_trajectory, order=order),
                )
                for order in range(max_derivative_order + 1)
            ]
        )
        np.minimum(reduced, stacked.mean(axis=0).min(axis=1), out=reduced)

    expected = pha.compute_min_distances(
        trajectory,
        small_rpos,
        max_derivative_order=max_derivative_order,
        downsample=downsample,
        n_jobs=1,
    )
    np.testing.assert_allclose(reduced, expected, rtol=1e-6, atol=1e-6)


def test_rescaled_min_distances_match_manual_scale_division(
    small_rpos: list[RPO],
    sample_initial_state: np.ndarray,
) -> None:
    """Dividing each order's Wasserstein matrix by the prepass scales before the
    order mean and phase/RPO minimum reproduces ``compute_min_distances`` with
    ``rescale_orders=True``."""
    max_derivative_order = 1
    downsample = 20
    trajectory = KSTrajectory.from_initial_state(
        sample_initial_state, dt=INTEGRATION_DT, num_timesteps=60, resolution=16
    )

    trajectory_diagrams = [
        KSPersistenceTrajectory.from_trajectory(trajectory, order=order)
        for order in range(max_derivative_order + 1)
    ]
    rpo_diagram_pairs = [
        (
            rpo,
            [
                KSPersistenceTrajectory.from_trajectory(
                    KSTrajectory.from_rpo(rpo, 16, downsample=downsample), order=order
                )
                for order in range(max_derivative_order + 1)
            ],
        )
        for rpo in small_rpos
    ]

    scales = _compute_order_scales(trajectory_diagrams, rpo_diagram_pairs)

    reduced = np.full(trajectory.num_timesteps, np.inf)
    for _, phase_diagrams_per_order in rpo_diagram_pairs:
        stacked = np.stack(
            [
                wasserstein_matrix(trajectory_diagrams[order], phase_diagrams_per_order[order])
                / scales[order]
                for order in range(max_derivative_order + 1)
            ]
        )
        np.minimum(reduced, stacked.mean(axis=0).min(axis=1), out=reduced)

    expected = pha.compute_min_distances(
        trajectory,
        small_rpos,
        max_derivative_order=max_derivative_order,
        downsample=downsample,
        rescale_orders=True,
        n_jobs=1,
    )
    np.testing.assert_allclose(reduced, expected, rtol=1e-6, atol=1e-6)
