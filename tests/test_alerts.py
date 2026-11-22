"""Frame-by-frame tests of the alert rules in alerts.py."""
import pytest

from alerts import AlertState, EarCalibrator

OPEN, CLOSED = 0.30, 0.10


def feed(state, frames, ear=OPEN, pitch=0.0, mar=0.2):
    return [state.update(ear, pitch, mar) for _ in range(frames)]


def test_calibrator_sets_threshold_to_75_percent_of_mean():
    cal = EarCalibrator(n_frames=4)
    assert [cal.add(v) for v in (0.28, 0.30, 0.32, 0.30)] == [False, False, False, True]
    assert cal.threshold == pytest.approx(0.75 * 0.30)
    cal.add(0.0)  # later values are ignored
    assert cal.threshold == pytest.approx(0.75 * 0.30)


def test_alert_only_after_consecutive_closed_frames():
    state = AlertState(eye_thresh=0.2, consec_frames=5)
    statuses = feed(state, 5, ear=CLOSED)
    assert [s.drowsy for s in statuses] == [False, False, False, False, True]


def test_opening_eyes_resets_the_count():
    state = AlertState(eye_thresh=0.2, consec_frames=5)
    feed(state, 4, ear=CLOSED)
    feed(state, 1, ear=OPEN)
    assert not any(s.drowsy for s in feed(state, 4, ear=CLOSED))


def test_head_down_counts_like_closed_eyes():
    state = AlertState(eye_thresh=0.2, pitch_thresh=15, consec_frames=3)
    assert feed(state, 3, pitch=20)[-1].drowsy
    assert not AlertState(eye_thresh=0.2, pitch_thresh=15, consec_frames=3).update(OPEN, 10, 0.2).drowsy


def test_yawn_needs_the_mouth_open_for_several_frames():
    state = AlertState(eye_thresh=0.2, mouth_thresh=0.6, yawn_frames=5)
    assert [s.yawn for s in feed(state, 5, mar=0.7)] == [False, False, False, False, True]
    assert not state.update(OPEN, 0, 0.5).yawn  # mouth closed again


def test_brief_mouth_opening_while_talking_is_not_a_yawn():
    state = AlertState(eye_thresh=0.2, mouth_thresh=0.6, yawn_frames=5)
    talking = [0.7, 0.3, 0.75, 0.4, 0.8, 0.2, 0.7, 0.7, 0.3]
    assert not any(state.update(OPEN, 0, m).yawn for m in talking)
