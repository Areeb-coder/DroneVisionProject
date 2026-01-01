import cv2
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import os

def run_drone_vision_test():
    # --- 1. SETUP MODELS (PUBLIC & ERROR-FREE) ---
    print("Initializing AI Models...")

    # Thermal Model: Still using the specialized heat signature model
    try:
        thermal_path = hf_hub_download(repo_id="pitangent-ds/YOLOv8-human-detection-thermal", filename="model.pt")
        model_thermal = YOLO(thermal_path)
    except Exception as e:
        print(f"Thermal download failed, using standard model as backup. Error: {e}")
        model_thermal = YOLO('yolov8n.pt')

    # Low-Light Model: Using standard YOLOv8s (Small) which is excellent for human detection
    # This downloads directly from Ultralytics servers (No 401 error possible)
    model_lowlight = YOLO('yolov8s.pt') 

    # --- 2. DEFINE YOUR INPUT VIDEOS ---
    # Put your video file names here (ensure they are in the same folder as this script)
    thermal_video_file = "a.mp4.mp4" 
    lowlight_video_file = "b.mp4.mp4"

    if not os.path.exists(thermal_video_file) or not os.path.exists(lowlight_video_file):
        print("Error: Video files not found! Please check the file names.")
        return

    cap_t = cv2.VideoCapture(thermal_video_file)
    cap_l = cv2.VideoCapture(lowlight_video_file)

    print("Processing Videos... Press 'q' to stop.")

    while cap_t.isOpened() and cap_l.isOpened():
        ret_t, frame_t = cap_t.read()
        ret_l, frame_l = cap_l.read()

        if not ret_t or not ret_l:
            break

        # --- 3. RUN DETECTION (DRAWING THE SQUARES) ---
        
        # Detect in Thermal Feed (GREEN SQUARES)
        res_t = model_thermal.predict(frame_t, conf=0.5, verbose=False)
        for box in res_t[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame_t, (x1, y1), (x2, y2), (0, 255, 0), 3) # Green box
            cv2.putText(frame_t, "HUMAN (HEAT)", (x1, y1-10), 0, 0.6, (0, 255, 0), 2)

        # Detect in Low-Light Feed (BLUE SQUARES)
        res_l = model_lowlight.predict(frame_l, conf=0.4, verbose=False)
        for box in res_l[0].boxes:
            # We filter for class 0 (Human/Person in the COCO dataset)
            if int(box.cls[0]) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame_l, (x1, y1), (x2, y2), (255, 0, 0), 3) # Blue box
                cv2.putText(frame_l, "HUMAN (VISUAL)", (x1, y1-10), 0, 0.6, (255, 0, 0), 2)

        # --- 4. DISPLAY THE DASHBOARD ---
        # Resize both to the same size for a neat side-by-side view
        out_t = cv2.resize(frame_t, (640, 480))
        out_l = cv2.resize(frame_l, (640, 480))
        combined = cv2.hconcat([out_t, out_l])

        cv2.imshow('Drone AI Test: Thermal (Green) | Low-Light (Blue)', combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_t.release()
    cap_l.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_drone_vision_test()