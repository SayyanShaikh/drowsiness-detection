"""Tests for detect_drowsiness.py that run without a webcam, dlib or a speaker."""
import pytest

import detect_drowsiness
from detect_drowsiness import DrowsinessDetector, require

DEFAULT_ARGS = {
    "shape_predictor": None, "camera": 0, "calibration_frames": 50, "alarm_cooldown": 10,
    "mouth_thresh": 0.6, "pitch_thresh": 15.0, "consec_frames": 20,
}


def test_module_imports_without_dlib_or_pyttsx3():
    assert hasattr(detect_drowsiness, "DrowsinessDetector")


def test_missing_dependency_gives_clear_message():
    with pytest.raises(ImportError, match="not installed. Run: pip install nothing-here"):
        require("module_that_does_not_exist_123", "Run: pip install nothing-here")


def test_detector_accepts_injected_models():
    fake_detector, fake_predictor = object(), object()
    d = DrowsinessDetector(dict(DEFAULT_ARGS), detector=fake_detector, predictor=fake_predictor)
    assert d.detector is fake_detector and d.predictor is fake_predictor
