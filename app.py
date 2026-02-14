import cv2
import time
import streamlit as st
from ultralytics import YOLO

# Load YOLO models
model_face = YOLO("yolov8n-face-lindevs.pt")   # face detection model
model_idcard = YOLO("best.pt")                 # ID card detection model

st.title("Real-Time Privacy Protection Demo")

# Ask user for their IP camera link
ip_link = st.text_input("Enter your IP camera link (or 0 for webcam)", "")

# Checkbox to start detection
run_demo = st.checkbox("Run Detection")

if run_demo and ip_link:
    cap = cv2.VideoCapture(ip_link if ip_link != "0" else 0)
    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("No video stream detected. Check your IP link.")
            break

        # Resize to 640x480 (good balance of detail + speed)
        frame = cv2.resize(frame, (640, 480))
        output = frame.copy()

        # --- Face detection ---
        results_face = model_face.predict(frame, conf=0.3, verbose=False)
        for r in results_face:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi = output[y1:y2, x1:x2]
                if roi.size > 0:
                    roi_blur = cv2.GaussianBlur(roi, (9, 9), 5)
                    output[y1:y2, x1:x2] = roi_blur
                cv2.putText(output, "Face", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # --- ID card detection ---
        results_id = model_idcard.predict(frame, conf=0.5, verbose=False)
        for r in results_id:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi = output[y1:y2, x1:x2]
                if roi.size > 0:
                    roi_blur = cv2.GaussianBlur(roi, (9, 9), 5)
                    output[y1:y2, x1:x2] = roi_blur
                cv2.putText(output, "ID Card", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Show frame in Streamlit
        stframe.image(output, channels="BGR")

        # Small sleep to reduce lag but keep accuracy
        time.sleep(0.01)

    cap.release()
