"""Tests for the face geometry in metrics.py (no webcam or dlib needed)."""
import cv2
import numpy as np
import pytest

from metrics import (MODEL_POINTS, POSE_LANDMARKS, camera_matrix, eye_aspect_ratio,
                     head_pitch, mouth_aspect_ratio, normalize_pitch)


def make_eye(height, width=30.0):
    """Six eye points (p1..p6) with the given opening height."""
    h = height / 2
    return np.array([[0, 0], [10, -h], [20, -h], [width, 0], [20, h], [10, h]], dtype=float)


def test_ear_of_open_and_closed_eye():
    assert eye_aspect_ratio(make_eye(9)) == pytest.approx(9 / 30)
    assert eye_aspect_ratio(make_eye(0)) == 0.0


def test_ear_drops_as_eye_closes():
    values = [eye_aspect_ratio(make_eye(h)) for h in (12, 9, 6, 3, 0)]
    assert values == sorted(values, reverse=True)


def test_ear_ignores_position_and_scale():
    eye = make_eye(8)
    moved = eye * 3.5 + [120, 45]
    assert eye_aspect_ratio(moved) == pytest.approx(eye_aspect_ratio(eye))


def test_mar_of_open_mouth():
    mouth = np.zeros((20, 2))
    mouth[0], mouth[6] = [0, 0], [60, 0]          # corners 48, 54
    mouth[2], mouth[10] = [20, -15], [20, 15]     # 50 and 58
    mouth[4], mouth[8] = [40, -15], [40, 15]      # 52 and 56
    assert mouth_aspect_ratio(mouth) == pytest.approx(30 / 60)


def test_bad_shapes_and_zero_width_are_handled():
    with pytest.raises(ValueError):
        eye_aspect_ratio(np.zeros((5, 2)))
    with pytest.raises(ValueError):
        mouth_aspect_ratio(np.zeros((12, 2)))
    assert eye_aspect_ratio(np.zeros((6, 2))) == 0.0
    assert mouth_aspect_ratio(np.zeros((20, 2))) == 0.0


@pytest.mark.parametrize("raw,expected", [(179.0, -1.0), (-170.0, 10.0), (-180.0, 0.0), (160.0, -20.0)])
def test_normalize_pitch(raw, expected):
    assert normalize_pitch(raw) == pytest.approx(expected)


def face_landmarks(pitch_deg, width=600, height=450):
    """Project the 3D face model with the head tilted down by `pitch_deg`."""
    a = np.deg2rad(pitch_deg)
    tilt = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    facing_camera = np.diag([1.0, -1.0, -1.0])  # model y up / z out -> image y down / z into scene
    rvec, _ = cv2.Rodrigues(facing_camera @ tilt)
    K = camera_matrix(width, height)
    pts, _ = cv2.projectPoints(MODEL_POINTS, rvec, np.array([[0.0], [0.0], [1000.0]]), K, np.zeros((4, 1)))
    shape = np.zeros((68, 2))
    shape[list(POSE_LANDMARKS)] = pts.reshape(-1, 2)
    return shape, K


@pytest.mark.parametrize("pitch", [-20, -5, 0, 10, 25])
def test_head_pitch_recovers_known_tilt(pitch):
    shape, K = face_landmarks(pitch)
    assert head_pitch(shape, K) == pytest.approx(pitch, abs=0.5)


def test_frontal_face_is_near_zero_not_180():
    shape, K = face_landmarks(0)
    assert abs(head_pitch(shape, K)) < 1.0
