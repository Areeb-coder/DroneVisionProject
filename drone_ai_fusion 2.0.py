import os
import cv2
import torch
from collections import defaultdict

from ultralytics import YOLO
from huggingface_hub import hf_hub_download


# =========================
# Decision Engine
# =========================
class DroneDecisionEngine:
    """
    Threat scoring per tracked ID:
    - Boundary line at y=350 on resized 640x480 frames.
    - +50 immediate when crossing into restricted zone (first time per entry).
    - +2 per frame while staying in restricted zone (loitering).
    - -1 per frame recovery when outside (min 0).
    Status:
      Safe (Green): score == 0
      Suspicious (Orange): 0 < score < 70
      Intruder (Red): score >= 70
    """

    def __init__(self, boundary_y: int = 350):
        self.boundary_y = boundary_y
        self.scores = defaultdict(int)         # id -> score
        self.in_zone = defaultdict(bool)       # id -> currently inside restricted zone

    def update(self, track_id: int, y2: int):
        """Update score for a single ID based on bbox base y2."""
        inside = y2 >= self.boundary_y

        # Crossing event: outside -> inside
        if inside and not self.in_zone[track_id]:
            self.scores[track_id] += 50

        # Loitering while inside restricted zone
        if inside:
            self.scores[track_id] += 2
        else:
            # Recovery when outside
            self.scores[track_id] = max(0, self.scores[track_id] - 1)

        # Update zone state
        self.in_zone[track_id] = inside

        # Determine status/color
        score = self.scores[track_id]
        if score == 0:
            status = "Safe"
            color = (0, 200, 0)  # green
        elif score < 70:
            status = "Suspicious"
            color = (0, 165, 255)  # orange
        else:
            status = "Intruder"
            color = (0, 0, 255)  # red

        return score, status, color


# =========================
# Visualization helpers
# =========================
def draw_boundary_line(frame, y=350, color=(255, 255, 255)):
    h, w = frame.shape[:2]
    cv2.line(frame, (0, y), (w, y), color, 2)
    cv2.putText(frame, f"Restricted boundary: y={y}", (10, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def put_mode_label(frame, label):
    cv2.rectangle(frame, (5, 5), (170, 35), (20, 20, 20), -1)
    cv2.putText(frame, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def overlay_status(frame, x1, y1, x2, y2, status, conf, score, color, track_id=None):
    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Status line: [Status] | AI Conf: [0.XX] | Threat Pts: [XX] | ID: [id]
    txt = f"[{status}] | AI Conf: {conf:.2f} | Threat Pts: {score}"
    if track_id is not None:
        txt += f" | ID: {track_id}"

    # Draw label background for readability
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    label_y = max(0, y1 - th - 6)
    cv2.rectangle(frame, (x1, label_y), (x1 + tw + 6, label_y + th + 6), (0, 0, 0), -1)
    cv2.putText(frame, txt, (x1 + 3, label_y + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


# =========================
# Model setup (High Precision)
# =========================
def load_models():
    print("Initializing High-Precision AI Models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        thermal_path = hf_hub_download(
            repo_id="pitangent-ds/YOLOv8-human-detection-thermal",
            filename="model.pt"
        )
        model_thermal = YOLO(thermal_path).to(device)
        print("Thermal model loaded from Hugging Face.")
    except Exception as e:
        print(f"Thermal download failed, fallback to yolov8l. Error: {e}")
        model_thermal = YOLO("yolov8l.pt").to(device)

    model_lowlight = YOLO("yolov8l.pt").to(device)
    print("Low-light model loaded with high precision.")
    return model_thermal, model_lowlight


# =========================
# Main application
# =========================
def open_video_sources(thermal_path="a.mp4.mov", lowlight_path="b.mp4.mov"):
    # Prefer local files; fallback to webcams
    cap_t = cv2.VideoCapture(thermal_path) if os.path.exists(thermal_path) else cv2.VideoCapture(0)
    cap_l = cv2.VideoCapture(lowlight_path) if os.path.exists(lowlight_path) else cv2.VideoCapture(1)

    if not cap_t.isOpened():
        print("Failed to open thermal source; attempting webcam index 0.")
        cap_t = cv2.VideoCapture(0)
    if not cap_l.isOpened():
        print("Failed to open low-light source; attempting webcam index 1.")
        cap_l = cv2.VideoCapture(1)

    if not cap_t.isOpened() or not cap_l.isOpened():
        raise RuntimeError("Unable to open both video sources (files or webcams).")

    return cap_t, cap_l


def process_streams():
    model_thermal, model_lowlight = load_models()

    # Tracker config (if not available, Ultralytics will fallback)
    tracker_cfg = "bytetrack.yaml"

    # High-Precision Parameters
    params = {
        "classes": [0],    # Humans only
        "conf": 0.70,      # Higher confidence threshold
        "imgsz": 1280,     # High resolution for better precision
        "iou": 0.40,       # Stricter overlap suppression
        "verbose": False
    }

    # Decision engines (per feed)
    decision_thermal = DroneDecisionEngine(boundary_y=350)
    decision_lowlight = DroneDecisionEngine(boundary_y=350)

    cap_t, cap_l = open_video_sources("a.mp4.mov", "b.mp4.mov")
    print("Processing streams... Press 'q' to stop.")

    while True:
        ret_t, frame_t = cap_t.read()
        ret_l, frame_l = cap_l.read()
        if not ret_t or not ret_l:
            break

        # Run tracking with persist IDs; fallback to predict if track fails
        try:
            res_t = model_thermal.track(source=frame_t, tracker=tracker_cfg, persist=True, **params)
        except Exception as e:
            print(f"Thermal track failed, fallback to predict. Error: {e}")
            res_t = model_thermal.predict(frame_t, **params)

        try:
            res_l = model_lowlight.track(source=frame_l, tracker=tracker_cfg, persist=True, **params)
        except Exception as e:
            print(f"Low-light track failed, fallback to predict. Error: {e}")
            res_l = model_lowlight.predict(frame_l, **params)

        # Extract first result (per frame)
        r_t = res_t[0] if len(res_t) > 0 else None
        r_l = res_l[0] if len(res_l) > 0 else None

        # Resize for consistent display and scoring boundary
        out_t = cv2.resize(frame_t, (640, 480))
        out_l = cv2.resize(frame_l, (640, 480))

        # Labels and boundary
        put_mode_label(out_t, "THERMAL")
        put_mode_label(out_l, "LOW-LIGHT")
        draw_boundary_line(out_t, y=decision_thermal.boundary_y)
        draw_boundary_line(out_l, y=decision_lowlight.boundary_y)

        # Thermal feed: filter for human class (ID=0)
        if r_t is not None and hasattr(r_t, "boxes") and r_t.boxes is not None and len(r_t.boxes) > 0:
            for b in r_t.boxes:
                cls_id = int(b.cls[0]) if getattr(b, "cls", None) is not None else 0
                if cls_id != 0:
                    continue

                x1, y1, x2, y2 = map(int, b.xyxy[0])
                conf = float(b.conf[0]) if getattr(b, "conf", None) is not None else 0.0
                track_id = int(b.id[0]) if getattr(b, "id", None) is not None else None

                if track_id is None:
                    score, status, color = (0, "Safe", (0, 200, 0))
                else:
                    score, status, color = decision_thermal.update(track_id, y2)

                overlay_status(out_t, x1, y1, x2, y2, status, conf, score, color, track_id)

        # Low-light feed: filter for human class (ID=0)
        if r_l is not None and hasattr(r_l, "boxes") and r_l.boxes is not None and len(r_l.boxes) > 0:
            for b in r_l.boxes:
                cls_id = int(b.cls[0]) if getattr(b, "cls", None) is not None else 0
                if cls_id != 0:
                    continue

                x1, y1, x2, y2 = map(int, b.xyxy[0])
                conf = float(b.conf[0]) if getattr(b, "conf", None) is not None else 0.0
                track_id = int(b.id[0]) if getattr(b, "id", None) is not None else None

                if track_id is None:
                    score, status, color = (0, "Safe", (0, 200, 0))
                else:
                    score, status, color = decision_lowlight.update(track_id, y2)

                overlay_status(out_l, x1, y1, x2, y2, status, conf, score, color, track_id)

        # Side-by-side display
        combined = cv2.hconcat([out_t, out_l])
        cv2.imshow("Drone Security: Thermal | Low-Light", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap_t.release()
    cap_l.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    process_streams()