from __future__ import annotations

import numpy as np
import trimesh.transformations as tf

from constants import (
    VIEWPORT_ORIENTATION_AXIS_ORIGIN,
    VIEWPORT_ORIENTATION_CUBE_SCALE,
)
from viewport import (
    _camera_up_vector,
    _interpolate_camera_pose,
    _scale_matrix_from_box,
    _slerp_unit_vectors,
    _transformed_bounds,
)


def test_orientation_arrows_start_at_the_scaled_cube_corner() -> None:
    expected_corner = np.full(3, 0.5 * VIEWPORT_ORIENTATION_CUBE_SCALE)

    assert np.allclose(VIEWPORT_ORIENTATION_AXIS_ORIGIN, expected_corner)


def test_camera_up_vector_stays_orthogonal_to_view_direction() -> None:
    direction = np.array([1.0, -2.0, -3.0])

    up = _camera_up_vector(direction)

    assert np.isclose(np.linalg.norm(up), 1.0)
    assert np.isclose(np.dot(up, direction), 0.0)
    assert np.dot(up, [0.0, 0.0, 1.0]) > 0.0


def test_camera_up_vector_uses_stable_fallback_when_looking_along_z() -> None:
    up = _camera_up_vector(np.array([0.0, 0.0, -1.0]))

    assert np.allclose(up, [0.0, 1.0, 0.0])


def test_camera_slerp_follows_a_smooth_shortest_arc() -> None:
    halfway = _slerp_unit_vectors(
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        0.5,
    )

    assert np.allclose(halfway, np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0))
    assert np.isclose(np.linalg.norm(halfway), 1.0)


def test_camera_pose_interpolation_keeps_an_orbit_radius() -> None:
    start = (
        np.array([10.0, 0.0, 0.0]),
        np.zeros(3),
        np.array([0.0, 0.0, 1.0]),
        12.0,
    )
    target = (
        np.array([0.0, 10.0, 0.0]),
        np.zeros(3),
        np.array([0.0, 0.0, 1.0]),
        20.0,
    )

    position, focal_point, up, parallel_scale = _interpolate_camera_pose(start, target, 0.5)

    assert np.allclose(focal_point, np.zeros(3))
    assert np.isclose(np.linalg.norm(position - focal_point), 10.0)
    assert np.allclose(position, np.array([1.0, 1.0, 0.0]) * (10.0 / np.sqrt(2.0)))
    assert np.allclose(up, np.array([0.0, 0.0, 1.0]))
    assert np.isclose(parallel_scale, 16.0)


def test_transformed_bounds_follow_model_matrix() -> None:
    bounds = np.array([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]])
    matrix = tf.translation_matrix([4.0, 5.0, 6.0]) @ np.diag([2.0, 3.0, 4.0, 1.0])

    minimum, maximum = _transformed_bounds(bounds, matrix)

    assert np.allclose(minimum, [2.0, -1.0, -6.0])
    assert np.allclose(maximum, [6.0, 11.0, 18.0])


def test_uniform_box_scale_keeps_rotation_and_proportions() -> None:
    rotation = tf.euler_matrix(0.2, 0.3, 0.4, axes="sxyz")
    baseline = tf.translation_matrix([1.0, 2.0, 3.0]) @ rotation @ np.diag([2.0, 3.0, 4.0, 1.0])
    candidate = tf.translation_matrix([1.5, 2.0, 3.0]) @ rotation @ np.diag([2.5, 3.0, 4.0, 1.0])

    result = _scale_matrix_from_box(candidate, baseline, uniform=True)
    scale, shear, angles, translation, _ = tf.decompose_matrix(result)

    assert np.allclose(scale, [2.5, 3.75, 5.0])
    assert np.allclose(shear, 0.0, atol=1e-9)
    assert np.allclose(angles, [0.2, 0.3, 0.4])
    assert np.allclose(translation, [1.5, 2.0, 3.0])


def test_box_scale_does_not_accept_inverted_model() -> None:
    baseline = np.eye(4)
    candidate = np.diag([-1.0, 1.0, 1.0, 1.0])

    result = _scale_matrix_from_box(candidate, baseline, uniform=False)

    assert np.allclose(result, baseline)
