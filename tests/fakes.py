"""Synthetic faces and fake dlib objects, so the detector can be tested without a camera."""
import cv2
import numpy as np

from metrics import MODEL_POINTS, POSE_LANDMARKS, camera_matrix

FRAME_W, FRAME_H = 600, 450


def synthetic_landmarks(eye_open=10.0, mouth_open=6.0, pitch_deg=0.0, offset=(0.0, 0.0), scale=1.0):
    """68 landmarks for a face looking at the camera.

    eye_open: eyelid gap in pixels (0 = closed); mouth_open: lip gap in pixels;
    pitch_deg: head tilted down by this many degrees.
    """
    a = np.deg2rad(pitch_deg)
    tilt = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    rvec, _ = cv2.Rodrigues(np.diag([1.0, -1.0, -1.0]) @ tilt)
    pts, _ = cv2.projectPoints(MODEL_POINTS, rvec, np.array([[0.0], [0.0], [1000.0 / scale]]),
                               camera_matrix(FRAME_W, FRAME_H), np.zeros((4, 1)))
    pts = pts.reshape(-1, 2)
    shape = np.zeros((68, 2))
    shape[list(POSE_LANDMARKS)] = pts

    def eye(outer_idx, direction):
        x0, y0 = shape[outer_idx]
        w, g = 40.0 * scale, eye_open * scale / 2
        xs = [x0, x0 + direction * w / 3, x0 + direction * 2 * w / 3, x0 + direction * w]
        return np.array([[xs[0], y0], [xs[1], y0 - g], [xs[2], y0 - g],
                         [xs[3], y0], [xs[2], y0 + g], [xs[1], y0 + g]])

    shape[36:42] = eye(36, +1)  # outer corner 36 -> inner corner 39
    left = eye(45, -1)          # outer corner 45 -> inner corner 42, stored in order 42..47
    shape[42:48] = left[[3, 2, 1, 0, 5, 4]]

    (lx, ly), (rx, ry) = shape[48], shape[54]
    gap = mouth_open * scale / 2
    xs = np.linspace(lx, rx, 7)
    outer_top = [(x, ly - gap) for x in xs[1:6]]            # 49-53
    outer_bottom = [(x, ry + gap) for x in xs[5:0:-1]]      # 55-59
    shape[49:54] = outer_top
    shape[55:60] = outer_bottom
    shape[60:68] = np.linspace(shape[48], shape[54], 8)     # inner lip (unused)
    return shape


class _Point:
    def __init__(self, x, y):
        self.x, self.y = int(round(x)), int(round(y))


class FakeShape:
    """Mimics dlib.full_object_detection, as read by imutils.face_utils.shape_to_np."""

    def __init__(self, landmarks):
        self._pts = landmarks
        self.num_parts = len(landmarks)

    def part(self, i):
        return _Point(*self._pts[i])


class FakeRect:
    def __init__(self, landmarks):
        self.landmarks = landmarks
        x0, y0 = landmarks.min(axis=0)
        x1, y1 = landmarks.max(axis=0)
        self._box = (x0, y0, x1, y1)

    def width(self):
        return self._box[2] - self._box[0]

    def height(self):
        return self._box[3] - self._box[1]


class FakeFaces:
    """Plays the role of both dlib's face detector and landmark predictor.

    Set `.faces` to a list of landmark arrays before each frame.
    """

    def __init__(self):
        self.faces = []

    def detect(self, gray, upsample=0):
        return [FakeRect(f) for f in self.faces]

    def predict(self, gray, rect):
        return FakeShape(rect.landmarks)


def blank_frame():
    return np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
