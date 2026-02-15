"""
Privacy Blur - BALANCED Version
Good speed + Good accuracy - Best of both worlds!
"""

from flask import Flask, render_template_string, Response, request, jsonify
import cv2
from ultralytics import YOLO
import numpy as np

app = Flask(__name__)

print("Loading models...")
model_face = YOLO("yolov8n-face-lindevs.pt")
model_idcard = YOLO("best.pt")
print("Models loaded!")

current_camera_url = None

# ADJUSTABLE SETTINGS - Change these for your needs!
PROCESS_EVERY_N_FRAMES = 2  # Process every 2nd frame (1=all frames, 2=every 2nd, 3=every 3rd)
RESOLUTION_WIDTH = 480      # 480 is balanced (320=fast, 640=accurate)
FACE_CONFIDENCE = 0.4       # 0.3=more detections, 0.6=fewer but accurate
ID_CONFIDENCE = 0.5         # Same as above
BLUR_STRENGTH = 17          # 11=light blur (fast), 25=heavy blur (slow)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Privacy Blur - Balanced Mode</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 {
            color: #667eea;
            text-align: center;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #f59e0b;
            font-weight: bold;
            margin-bottom: 30px;
        }
        .settings-box {
            background: #fef3c7;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            border-left: 5px solid #f59e0b;
        }
        .settings-box h3 {
            color: #92400e;
            margin-bottom: 10px;
        }
        .setting-item {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            color: #78350f;
        }
        .setting-value {
            font-weight: bold;
            color: #f59e0b;
        }
        .input-section {
            background: #f8f9ff;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
        }
        .input-group {
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
        }
        input {
            flex: 1;
            padding: 15px;
            border: 2px solid #e0e7ff;
            border-radius: 10px;
            font-size: 16px;
        }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            border-radius: 10px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        .stop-btn {
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        }
        .video-container {
            display: none;
            text-align: center;
        }
        .video-container.active {
            display: block;
        }
        img {
            max-width: 100%;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .info-box {
            background: #e0e7ff;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }
        .info-box h3 {
            color: #667eea;
            margin-bottom: 10px;
        }
        .status {
            background: #10b981;
            color: white;
            padding: 10px 20px;
            border-radius: 8px;
            display: inline-block;
            margin-top: 10px;
        }
        .example {
            background: white;
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
            font-family: monospace;
            color: #4c1d95;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔒 Privacy Blur - Balanced Mode</h1>
        <div class="subtitle">⚖️ Optimized for Speed + Accuracy</div>
        
        <div class="settings-box">
            <h3>⚙️ Current Settings:</h3>
            <div class="setting-item">
                <span>Frame Processing:</span>
                <span class="setting-value">Every 2nd frame</span>
            </div>
            <div class="setting-item">
                <span>Resolution:</span>
                <span class="setting-value">480x360 (Balanced)</span>
            </div>
            <div class="setting-item">
                <span>Face Detection:</span>
                <span class="setting-value">Confidence 40% (Good accuracy)</span>
            </div>
            <div class="setting-item">
                <span>Blur Quality:</span>
                <span class="setting-value">Medium (Fast + Effective)</span>
            </div>
            <div style="margin-top: 10px; font-size: 14px; color: #78350f;">
                💡 Want to adjust? Edit the settings at the top of the Python file!
            </div>
        </div>
        
        <div class="input-section">
            <h2 style="color: #667eea; margin-bottom: 20px;">Enter Your IP Camera URL</h2>
            
            <div class="input-group">
                <input type="text" id="cameraUrl" placeholder="http://192.168.1.100:8080/video" 
                       value="http://172.16.196.111:8080/video">
                <button onclick="startCamera()">🚀 Start Stream</button>
            </div>
            
            <div style="background: #e0e7ff; padding: 15px; border-radius: 8px;">
                <strong>📱 Example URL formats:</strong>
                <div class="example">http://192.168.1.100:8080/video</div>
                <div class="example">http://your-phone-ip:8080/video</div>
            </div>
        </div>

        <div class="video-container" id="videoContainer">
            <button class="stop-btn" onclick="stopCamera()">⏹️ Stop Stream</button>
            
            <div style="margin-top: 20px;">
                <img id="videoStream" src="/video_feed" style="display: none;">
            </div>
            
            <div class="info-box">
                <h3>🎯 Detection Active:</h3>
                <p>✅ <strong style="color: #10b981;">Green box:</strong> Main speaker (stays clear)</p>
                <p>🔴 <strong style="color: #ef4444;">Red box:</strong> Background people (blurred)</p>
                <p>🔵 <strong style="color: #3b82f6;">Blue box:</strong> ID cards (blurred)</p>
                <div class="status" id="status">Ready to connect</div>
            </div>
        </div>
    </div>

    <script>
        function startCamera() {
            const url = document.getElementById('cameraUrl').value;
            
            if (!url) {
                alert('Please enter a camera URL!');
                return;
            }

            fetch('/set_camera', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({url: url})
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    document.getElementById('videoContainer').classList.add('active');
                    document.getElementById('videoStream').style.display = 'block';
                    document.getElementById('videoStream').src = '/video_feed?' + new Date().getTime();
                    document.getElementById('status').textContent = '🟢 Live - Balanced Mode';
                    document.getElementById('status').style.background = '#10b981';
                } else {
                    alert('Error: ' + data.message);
                }
            })
            .catch(error => {
                alert('Connection failed: ' + error);
            });
        }

        function stopCamera() {
            fetch('/stop_camera', {method: 'POST'})
            .then(() => {
                document.getElementById('videoContainer').classList.remove('active');
                document.getElementById('videoStream').style.display = 'none';
                document.getElementById('status').textContent = 'Stopped';
                document.getElementById('status').style.background = '#6b7280';
            });
        }
    </script>
</body>
</html>
"""

def process_frame(frame):
    """Balanced processing - good speed + good accuracy"""
    output = frame.copy()
    
    # Face detection with balanced confidence
    results_face = model_face.predict(
        source=frame, 
        conf=FACE_CONFIDENCE,  # 0.4 = balanced
        verbose=False, 
        imgsz=RESOLUTION_WIDTH
    )
    face_boxes = []
    
    for r in results_face:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face_boxes.append((x1, y1, x2, y2))
    
    # Find largest face
    largest_box = None
    if face_boxes:
        largest_box = max(face_boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
    
    # Process faces
    for (x1, y1, x2, y2) in face_boxes:
        if largest_box and (x1, y1, x2, y2) == largest_box:
            # Main speaker - green box
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(output, "Speaker", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            # Background - blur with balanced strength
            roi = output[y1:y2, x1:x2]
            if roi.size > 0:
                # Gaussian blur with balanced kernel
                roi_blur = cv2.GaussianBlur(roi, (BLUR_STRENGTH, BLUR_STRENGTH), 15)
                output[y1:y2, x1:x2] = roi_blur
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(output, "Person", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # ID card detection with balanced confidence
    results_id = model_idcard.predict(
        source=frame,
        conf=ID_CONFIDENCE,  # 0.5 = balanced
        verbose=False,
        imgsz=RESOLUTION_WIDTH
    )
    
    for r in results_id:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            area = w * h
            
            # Skip unrealistic sizes
            if area < 800 or area > 80000:
                continue
            
            # Blur ID card with strong blur
            roi = output[y1:y2, x1:x2]
            if roi.size > 0:
                roi_blur = cv2.GaussianBlur(roi, (31, 31), 30)  # Strong blur for IDs
                output[y1:y2, x1:x2] = roi_blur
            
            cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(output, f"ID Card", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return output

def generate_frames():
    """Balanced frame generation"""
    global current_camera_url
    
    if not current_camera_url:
        return
    
    cap = cv2.VideoCapture(current_camera_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    
    if not cap.isOpened():
        print(f"Error: Cannot open camera URL: {current_camera_url}")
        return
    
    print(f"Connected to camera: {current_camera_url}")
    frame_count = 0
    
    try:
        while current_camera_url:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every Nth frame based on settings
            if frame_count % PROCESS_EVERY_N_FRAMES != 0:
                continue
            
            # Resize to balanced resolution
            height = int(RESOLUTION_WIDTH * 3 / 4)  # Maintain 4:3 aspect ratio
            frame = cv2.resize(frame, (RESOLUTION_WIDTH, height))
            
            # Process with AI
            output = process_frame(frame)
            
            # Encode with good quality
            _, buffer = cv2.imencode('.jpg', output, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
    finally:
        cap.release()
        print("Camera released")

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/set_camera', methods=['POST'])
def set_camera():
    global current_camera_url
    data = request.json
    url = data.get('url', '')
    
    if not url:
        return jsonify({'status': 'error', 'message': 'No URL provided'})
    
    test_cap = cv2.VideoCapture(url)
    if not test_cap.isOpened():
        test_cap.release()
        return jsonify({'status': 'error', 'message': 'Cannot connect to camera'})
    test_cap.release()
    
    current_camera_url = url
    print(f"Camera URL set to: {url}")
    
    return jsonify({'status': 'success'})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global current_camera_url
    current_camera_url = None
    return jsonify({'status': 'success'})

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("\n" + "="*60)
    print("⚖️  Privacy Blur - BALANCED Mode")
    print("="*60)
    print("\n📊 Settings:")
    print(f"   • Process every {PROCESS_EVERY_N_FRAMES} frames")
    print(f"   • Resolution: {RESOLUTION_WIDTH}x{int(RESOLUTION_WIDTH*3/4)}")
    print(f"   • Face confidence: {FACE_CONFIDENCE*100}%")
    print(f"   • ID confidence: {ID_CONFIDENCE*100}%")
    print(f"   • Blur strength: {BLUR_STRENGTH}")
    print("\n✅ Server starting...")
    print("📱 Open: http://localhost:5000")
    print("\n💡 To adjust settings, edit the variables at the top of this file")
    print("\n⏹️  Press Ctrl+C to stop\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
