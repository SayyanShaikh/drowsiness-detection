"""Detect drowsiness using facial landmarks, EAR/MAR and head pose.

Usage: python detect_drowsiness.py --shape-predictor path/to/shape_predictor_68_face_landmarks.dat

This module captures webcam frames, computes Eye Aspect Ratio (EAR), Mouth
Aspect Ratio (MAR), and a head-pose pitch angle to detect signs of drowsiness.
When threshold conditions are met it triggers a text-to-speech alert.
"""

import argparse
import time
import dlib
import cv2
import pyttsx3
import imutils
import threading
import logging
import os
import numpy as np
from imutils.video import VideoStream
from imutils import face_utils

from metrics import camera_matrix, eye_aspect_ratio, head_pitch, mouth_aspect_ratio

class DrowsinessDetector:
    def __init__(self, args):
        self.args = args
        self.initialize_logging()
        
        # --- Thresholds & State ---
        self.eye_thresh = 0.2  # Will be updated by calibration
        self.counter = 0
        self.alarm_on = False
        self.last_alarm_time = 0.0
        
        # --- Calibration State ---
        self.cal_ear_values = []
        self.calibration_done = False
        
        # --- Load Models ---
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = self.load_predictor()
        
        # --- Landmark Indices ---
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (self.mStart, self.mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    def initialize_logging(self):
        logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    def load_predictor(self):
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
            engine = pyttsx3.init()
            engine.setProperty('rate', 145)
            engine.say(message)
            engine.runAndWait()
        
        # Only trigger if cooldown period has passed
        if (time.time() - self.last_alarm_time) > self.args["alarm_cooldown"]:
            self.last_alarm_time = time.time()
            threading.Thread(target=speak, daemon=True).start()

    def run(self):
        """Main processing loop."""
        logging.info("Starting Video Stream...")
        vs = VideoStream(src=self.args["camera"]).start()
        time.sleep(2.0)
        
        cam_matrix = None

        try:
            while True:
                frame = vs.read()
                if frame is None: continue
                
                frame = imutils.resize(frame, width=600)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Initialize camera matrix once we know the frame dimensions
                if cam_matrix is None:
                    h, w = frame.shape[:2]
                    cam_matrix = camera_matrix(w, h)

                rects = self.detector(gray, 0)

                for rect in rects:
                    shape = face_utils.shape_to_np(self.predictor(gray, rect))

                    # 1. EAR Logic
                    leftEye = shape[self.lStart:self.lEnd]
                    rightEye = shape[self.rStart:self.rEnd]
                    ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0

                    # Calibration logic
                    if not self.calibration_done:
                        self.cal_ear_values.append(ear)
                        cv2.putText(frame, f"Calibrating: {len(self.cal_ear_values)}/{self.args['calibration_frames']}", 
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        if len(self.cal_ear_values) >= self.args["calibration_frames"]:
                            self.eye_thresh = np.mean(self.cal_ear_values) * 0.75
                            self.calibration_done = True
                            logging.info(f"Calibration finished. Threshold set to: {self.eye_thresh:.2f}")
                        continue

                    # 2. MAR Logic (Yawning)
                    mar = mouth_aspect_ratio(shape[self.mStart:self.mEnd])

                    # 3. Head Pose Logic (Nodding)
                    pitch = head_pitch(shape, cam_matrix)

                    # Visual Overlays
                    for eye in [leftEye, rightEye]:
                        cv2.drawContours(frame, [cv2.convexHull(eye)], -1, (0, 255, 0), 1)

                    # Trigger Logic
                    if ear < self.eye_thresh or pitch > self.args["pitch_thresh"]:
                        self.counter += 1
                        if self.counter >= self.args["consec_frames"]:
                            cv2.putText(frame, "DANGER: WAKE UP!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                            self.play_alert("Attention. Please stay alert.")
                    else:
                        self.counter = 0

                    if mar > self.args["mouth_thresh"]:
                        cv2.putText(frame, "YAWN DETECTED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                    # On-screen Telemetry
                    cv2.putText(frame, f"EAR: {ear:.2f} | Pitch: {pitch:.1f}", (10, frame.shape[0]-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                cv2.imshow("Safety System", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cv2.destroyAllWindows()
            vs.stop()

if __name__ == "__main__":
    # Setup Argument Parsing
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--shape-predictor", help="Path to landmark predictor")
    ap.add_argument("-c", "--camera", type=int, default=0, help="Webcam index")
    ap.add_argument("--calibration-frames", type=int, default=50)
    ap.add_argument("--alarm-cooldown", type=int, default=10)
    ap.add_argument("--mouth-thresh", type=float, default=0.6)
    ap.add_argument("--pitch-thresh", type=float, default=15)
    ap.add_argument("--consec-frames", type=int, default=20)
    
    args = vars(ap.parse_args())
    detector_system = DrowsinessDetector(args)
    detector_system.run()