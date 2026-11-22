"""Face geometry used to judge drowsiness from dlib's 68 facial landmarks.

Landmark numbers follow the iBUG 300-W 68-point scheme used by dlib:

* 36-41 and 42-47: the two eyes (6 points each)
* 48-59: outer edge of the lips, 60-67: inner edge
* 30: nose tip, 8: chin, 36/45: outer eye corners, 48/54: mouth corners

Only NumPy is needed here, except `head_pitch`, which uses OpenCV.
"""
import numpy as np

# Generic 3D face model (millimetres) for the six points used in head-pose estimation:
# nose tip, chin, left eye outer corner, right eye outer corner, left and right mouth corners.
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0),
])
POSE_LANDMARKS = (30, 8, 36, 45, 48, 54)


def _dist(a, b):
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def eye_aspect_ratio(eye):
    """Eye Aspect Ratio (Soukupova & Cech, 2016) for the 6 points of one eye.

    EAR = (|p2 - p6| + |p3 - p5|) / (2 |p1 - p4|)

    It is roughly constant while the eye is open and drops towards 0 as it closes.
    """
    eye = np.asarray(eye, dtype=float)
    if eye.shape != (6, 2):
        raise ValueError(f"expected 6 eye points, got shape {eye.shape}")
    width = _dist(eye[0], eye[3])
    if width == 0:
        return 0.0
    return (_dist(eye[1], eye[5]) + _dist(eye[2], eye[4])) / (2.0 * width)


def mouth_aspect_ratio(mouth):
    """Mouth Aspect Ratio for the 20 mouth points (landmarks 48-67).

    Uses the outer lip: two vertical distances (landmarks 50-58 and 52-56)
    divided by twice the mouth width (landmarks 48-54).
    """
    mouth = np.asarray(mouth, dtype=float)
    if mouth.shape != (20, 2):
        raise ValueError(f"expected 20 mouth points, got shape {mouth.shape}")
    width = _dist(mouth[0], mouth[6])
    if width == 0:
        return 0.0
    return (_dist(mouth[2], mouth[10]) + _dist(mouth[4], mouth[8])) / (2.0 * width)


def normalize_pitch(raw_pitch):
    """Convert OpenCV's raw pitch so that 0 = facing the camera and positive = head down.

    The 3D model's y-axis points up but image y points down, so a face looking
    straight at the camera comes out near +/-180 degrees.
    """
    return raw_pitch - 180 if raw_pitch > 0 else raw_pitch + 180


def camera_matrix(width, height):
    """Approximate pinhole camera: focal length = image width, centre = image centre."""
    return np.array([
        [width, 0, width / 2],
        [0, width, height / 2],
        [0, 0, 1],
    ], dtype="double")


def head_pitch(shape, cam_matrix):
    """Head pitch in degrees (positive = tilted down) from the 68 landmarks."""
    import cv2

    image_points = np.asarray(shape, dtype="double")[list(POSE_LANDMARKS)]
    _, rot_vec, trans_vec = cv2.solvePnP(MODEL_POINTS, image_points, cam_matrix, np.zeros((4, 1)))
    rmat, _ = cv2.Rodrigues(rot_vec)
    raw = cv2.decomposeProjectionMatrix(np.hstack((rmat, trans_vec)))[6].flatten()[0]
    return float(normalize_pitch(raw))
