from __future__ import annotations

import numpy as np
import trimesh.transformations as tf

from viewport import _scale_matrix_from_box, _transformed_bounds


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
