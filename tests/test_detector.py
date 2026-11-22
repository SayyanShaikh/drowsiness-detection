"""Tests for detect_drowsiness.py that run without a webcam, dlib or a speaker."""
import pytest
from fakes import FakeFaces, blank_frame, synthetic_landmarks

import detect_drowsiness
from detect_drowsiness import DrowsinessDetector, parse_args, require


def make_detector(**overrides):
    args = parse_args([])
    args.update(calibration_frames=5, consec_frames=4, **overrides)
    faces = FakeFaces()
    det = DrowsinessDetector(args, detector=faces.detect, predictor=faces.predict)
    det.spoken = []
    det.play_alert = det.spoken.append  # record alerts instead of speaking
    return det, faces


def run_frames(det, faces, n, landmarks):
    statuses = []
    for _ in range(n):
        faces.faces = [] if landmarks is None else [landmarks]
        _, status = det.process_frame(blank_frame())
        statuses.append(status)
    return statuses


def test_module_imports_without_dlib_or_pyttsx3():
    assert hasattr(detect_drowsiness, "DrowsinessDetector")


def test_missing_dependency_gives_clear_message():
    with pytest.raises(ImportError, match="not installed. Run: pip install nothing-here"):
        require("module_that_does_not_exist_123", "Run: pip install nothing-here")


def test_detector_accepts_injected_models():
    det, faces = make_detector()
    assert det.detector == faces.detect and det.predictor == faces.predict


def test_calibrates_then_alerts_on_closed_eyes():
    det, faces = make_detector()
    run_frames(det, faces, 5, synthetic_landmarks(eye_open=10))
    assert det.calibrator.done and det.calibrator.threshold == pytest.approx(0.75 * 0.25, abs=0.01)
    assert not any(s.drowsy for s in run_frames(det, faces, 3, synthetic_landmarks(eye_open=0)))
    assert run_frames(det, faces, 1, synthetic_landmarks(eye_open=0))[-1].drowsy
    assert det.spoken == ["Attention. Please stay alert."]


def test_head_nod_triggers_alert_with_open_eyes():
    det, faces = make_detector()
    run_frames(det, faces, 5, synthetic_landmarks())
    assert run_frames(det, faces, 4, synthetic_landmarks(pitch_deg=25))[-1].drowsy


def test_open_eyes_and_level_head_never_alert():
    det, faces = make_detector()
    statuses = run_frames(det, faces, 30, synthetic_landmarks())
    assert not any(s.drowsy for s in statuses) and det.spoken == []
