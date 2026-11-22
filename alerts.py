"""Decide when to raise alerts from per-frame face measurements.

Kept separate from the webcam loop so the rules can be tested frame by frame.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class FrameStatus:
    """What one frame means for the driver."""

    drowsy: bool = False  # eyes closed or head down for long enough -> voice alert
    yawn: bool = False


class EarCalibrator:
    """Learns a person's normal open-eye EAR over the first frames.

    The closed-eye threshold is `ratio` times their average, so the detector adapts
    to different eye shapes instead of using one fixed number.
    """

    def __init__(self, n_frames=50, ratio=0.75):
        self.n_frames = n_frames
        self.ratio = ratio
        self.values = []
        self.threshold = None

    @property
    def done(self):
        return self.threshold is not None

    def add(self, ear):
        """Record one EAR value; returns True once calibration is complete."""
        if not self.done:
            self.values.append(float(ear))
            if len(self.values) >= self.n_frames:
                self.threshold = float(np.mean(self.values)) * self.ratio
        return self.done


class AlertState:
    """Counts consecutive warning frames and reports when an alert is due."""

    def __init__(self, eye_thresh, pitch_thresh=15.0, consec_frames=20, mouth_thresh=0.6):
        self.eye_thresh = eye_thresh
        self.pitch_thresh = pitch_thresh
        self.consec_frames = consec_frames
        self.mouth_thresh = mouth_thresh
        self.counter = 0

    def update(self, ear, pitch, mar):
        """Feed one frame's measurements; returns the FrameStatus for that frame."""
        if ear < self.eye_thresh or pitch > self.pitch_thresh:
            self.counter += 1
        else:
            self.counter = 0
        return FrameStatus(drowsy=self.counter >= self.consec_frames, yawn=mar > self.mouth_thresh)
