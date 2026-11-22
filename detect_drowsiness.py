"""Detect drowsiness using facial landmarks, EAR/MAR and head pose.

Usage: python detect_drowsiness.py --shape-predictor path/to/shape_predictor_68_face_landmarks.dat

This module captures webcam frames, computes Eye Aspect Ratio (EAR), Mouth
Aspect Ratio (MAR), and a head-pose pitch angle to detect signs of drowsiness.
When threshold conditions are met it triggers a text-to-speech alert.
"""

import argparse
import collections
import importlib
import logging
import os
import threading
import time

import cv2
import imutils
from imutils import face_utils
from imutils.video import VideoStream

from alerts import AlertState, EarCalibrator, FrameStatus
from metrics import camera_matrix, eye_aspect_ratio, head_pitch, mouth_aspect_ratio


def require(module_name, install_hint):
    """Import an optional heavy dependency, with a clear message if it is missing."""
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(f"'{module_name}' is not installed. {install_hint}") from exc


class FpsMeter:
    """Frames per second, smoothed over the last `window` frames."""

    def __init__(self, window=30, clock=time.perf_counter):
        self.times = collections.deque(maxlen=window)
        self.clock = clock

    def tick(self):
        """Call once per frame; returns the current FPS (0 until two frames are seen)."""
        self.times.append(self.clock())
        if len(self.times) < 2 or self.times[-1] == self.times[0]:
            return 0.0
        return (len(self.times) - 1) / (self.times[-1] - self.times[0])


class FileStream:
    """Frames from a video file, with the same start/read/stop methods as imutils' VideoStream."""

    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {path}")
        self.finished = False

    def start(self):
        return self

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            self.finished = True
            return None
        return frame

    def stop(self):
        self.cap.release()


class DrowsinessDetector:
    MAX_EMPTY_READS = 50  # about 5 seconds without a frame -> give up with a clear message

    def __init__(self, args, detector=None, predictor=None):
        """`detector` and `predictor` can be passed in (e.g. fakes in tests);
        by default dlib's face detector and the 68-landmark model are loaded."""
        self.args = args
        self.initialize_logging()

        # --- State ---
        self.calibrator = EarCalibrator(n_frames=args["calibration_frames"])
        self.alerts = None  # created once calibration has set the eye threshold
        self.cam_matrix = None
        self.last_alarm_time = 0.0

        # --- Load Models ---
        if detector is None or predictor is None:
            dlib = require("dlib", "Run: pip install dlib  (see README for Windows notes)")
            self.detector = detector or dlib.get_frontal_face_detector()
            self.predictor = predictor or self.load_predictor(dlib)
        else:
            self.detector, self.predictor = detector, predictor

        # --- Landmark Indices ---
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (self.mStart, self.mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    def initialize_logging(self):
        logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    def load_predictor(self, dlib):
        """Attempts to find and load the dlib facial landmark predictor."""
        path = self.args["shape_predictor"]
        if not path:
            candidates = ["shape_predictor_68_face_landmarks.dat", "shape_predictor.dat"]
            for c in candidates:
                p = os.path.join(os.path.dirname(__file__), c)
                if os.path.exists(p):
                    path = p
                    break

        if not path or not os.path.exists(path):
            raise FileNotFoundError("Shape predictor file not found. Download it from dlib.net.")

        logging.info(f"Loading predictor from: {path}")
        return dlib.shape_predictor(path)

    def play_alert(self, message):
        """Triggers the TTS engine in a background thread."""
        def speak():
            pyttsx3 = require("pyttsx3", "Run: pip install pyttsx3")
            engine = pyttsx3.init()
            engine.setProperty('rate', 145)
            engine.say(message)
            engine.runAndWait()

        # Only trigger if cooldown period has passed
        if (time.time() - self.last_alarm_time) > self.args["alarm_cooldown"]:
            self.last_alarm_time = time.time()
            threading.Thread(target=speak, daemon=True).start()

    def process_frame(self, frame):
        """Analyse one BGR frame. Returns (frame with overlays drawn, FrameStatus)."""
        frame = imutils.resize(frame, width=600)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Initialize camera matrix once we know the frame dimensions
        if self.cam_matrix is None:
            h, w = frame.shape[:2]
            self.cam_matrix = camera_matrix(w, h)

        status = FrameStatus()
        rects = self.detector(gray, 0)
        if len(rects) == 0:
            if self.alerts is not None:
                status = self.alerts.update_no_face()
                if status.face_lost:
                    cv2.putText(frame, "FACE NOT VISIBLE!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    self.play_alert("Face not visible. Please stay alert.")
            return frame, status

        # With several faces in view (e.g. a passenger), follow the largest one:
        # the driver is the person closest to the camera.
        rect = max(rects, key=lambda r: r.width() * r.height())
        shape = face_utils.shape_to_np(self.predictor(gray, rect))

        # 1. EAR Logic
        left_eye = shape[self.lStart:self.lEnd]
        right_eye = shape[self.rStart:self.rEnd]
        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

        # Calibration logic
        if not self.calibrator.done:
            self.calibrator.add(ear)
            cv2.putText(frame, f"Calibrating: {len(self.calibrator.values)}/{self.calibrator.n_frames}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if self.calibrator.done:
                self.alerts = AlertState(
                    eye_thresh=self.calibrator.threshold,
                    pitch_thresh=self.args["pitch_thresh"],
                    consec_frames=self.args["consec_frames"],
                    mouth_thresh=self.args["mouth_thresh"],
                    yawn_frames=self.args["yawn_frames"],
                    lost_frames=self.args["lost_frames"],
                )
                logging.info(f"Calibration finished. Threshold set to: {self.calibrator.threshold:.2f}")
            return frame, status

        # 2. MAR Logic (Yawning) and 3. Head Pose Logic (Nodding)
        mar = mouth_aspect_ratio(shape[self.mStart:self.mEnd])
        pitch = head_pitch(shape, self.cam_matrix)

        # Visual Overlays
        for eye in [left_eye, right_eye]:
            cv2.drawContours(frame, [cv2.convexHull(eye)], -1, (0, 255, 0), 1)

        # Trigger Logic
        status = self.alerts.update(ear, pitch, mar)
        if status.drowsy:
            cv2.putText(frame, "DANGER: WAKE UP!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            self.play_alert("Attention. Please stay alert.")
        if status.yawn:
            cv2.putText(frame, "YAWN DETECTED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # On-screen Telemetry
        cv2.putText(frame, f"EAR: {ear:.2f} | Pitch: {pitch:.1f}", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return frame, status

    def run(self):
        """Main processing loop."""
        if self.args.get("video"):
            logging.info(f"Reading video file: {self.args['video']}")
            vs = FileStream(self.args["video"]).start()
        else:
            logging.info("Starting Video Stream...")
            vs = VideoStream(src=self.args["camera"]).start()
            time.sleep(2.0)  # let the camera warm up

        show_window = not self.args.get("no_window", False)
        self.frames_processed = 0
        self.fps = 0.0
        fps_meter = FpsMeter()
        empty_reads = 0
        try:
            while True:
                frame = vs.read()
                if frame is None:
                    if getattr(vs, "finished", False):
                        logging.info(f"End of video after {self.frames_processed} frames "
                                     f"(processed at {self.fps:.1f} FPS).")
                        break
                    empty_reads += 1
                    if empty_reads >= self.MAX_EMPTY_READS:
                        raise RuntimeError(
                            f"No video from camera {self.args['camera']}. Check that it is connected and not "
                            "in use by another program, or try another index with --camera.")
                    time.sleep(0.1)
                    continue
                empty_reads = 0

                frame, _ = self.process_frame(frame)
                self.frames_processed += 1
                self.fps = fps_meter.tick()
                cv2.putText(frame, f"FPS: {self.fps:.1f}", (frame.shape[1] - 85, frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                if show_window:
                    cv2.imshow("Safety System", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            if show_window:
                cv2.destroyAllWindows()
            vs.stop()


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--shape-predictor", help="Path to landmark predictor")
    ap.add_argument("-c", "--camera", type=int, default=0, help="Webcam index")
    ap.add_argument("-v", "--video", help="Analyse a recorded video file instead of the webcam")
    ap.add_argument("--no-window", action="store_true", help="Do not open a preview window")
    ap.add_argument("--calibration-frames", type=int, default=50)
    ap.add_argument("--alarm-cooldown", type=int, default=10)
    ap.add_argument("--mouth-thresh", type=float, default=0.6)
    ap.add_argument("--yawn-frames", type=int, default=15,
                    help="Frames in a row with the mouth wide open before a yawn is reported")
    ap.add_argument("--pitch-thresh", type=float, default=15)
    ap.add_argument("--consec-frames", type=int, default=20)
    ap.add_argument("--lost-frames", type=int, default=40,
                    help="Frames in a row with no face found (after calibration) before an alert")
    return vars(ap.parse_args(argv))


def main(argv=None):
    try:
        DrowsinessDetector(parse_args(argv)).run()
    except (RuntimeError, FileNotFoundError, ImportError) as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
