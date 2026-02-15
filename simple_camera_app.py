"""
Privacy Blur - Simple IP Camera Web App
Just run this file and open http://localhost:5000 in your browser!
"""

from flask import Flask, render_template_string, Response, request, jsonify
import cv2
from ultralytics import YOLO
import numpy as np

app = Flask(__name__)

# Load your YOLO models (make sure these files are in the same folder!)
print("Loading models...")
model_face = YOLO("yolov8n-face-lindevs.pt")
model_idcard = YOLO("best.pt")
print("Models loaded!")

# Global variable to store camera URL
current_camera_url = None

# HTML Template - The entire website in one string!
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Privacy Blur - IP Camera</title>
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
            margin-bottom: 30px;
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
        <h1>🔒 Privacy Blur - IP Camera Live Stream</h1>
        
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
                <h3>🎯 What's Being Detected:</h3>
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

            // Send URL to backend
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
                    document.getElementById('status').textContent = '🟢 Live - Processing frames...';
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
    """Process frame with face and ID card detection"""
    output = frame.copy()
    
    # Detect faces
    results_face = model_face.predict(source=frame, conf=0.3, verbose=False)
    face_boxes = []
    
    for r in results_face:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face_boxes.append((x1, y1, x2, y2))
    
    # Find largest face (main speaker)
    largest_box = None
    if face_boxes:
        largest_box = max(face_boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
    
    # Process faces
    for (x1, y1, x2, y2) in face_boxes:
        if largest_box and (x1, y1, x2, y2) == largest_box:
            # Main speaker - green box
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(output, "Speaker", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            # Background person - blur and red box
            roi = output[y1:y2, x1:x2]
            if roi.size > 0:
                roi_blur = cv2.GaussianBlur(roi, (23, 23), 30)
                output[y1:y2, x1:x2] = roi_blur
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(output, "Person", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Detect ID cards
    results_id = model_idcard.predict(source=frame, conf=0.5, verbose=False)
    
    for r in results_id:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            area = w * h
            
            # Skip unrealistic sizes
            if area < 1000 or area > 100000:
                continue
            
            # Blur ID card
            roi = output[y1:y2, x1:x2]
            if roi.size > 0:
                roi_blur = cv2.GaussianBlur(roi, (51, 51), 50)
                output[y1:y2, x1:x2] = roi_blur
            
            cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(output, f"ID Card", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    return output

def generate_frames():
    """Generate video frames from IP camera"""
    global current_camera_url
    
    if not current_camera_url:
        return
    
    cap = cv2.VideoCapture(current_camera_url)
    
    if not cap.isOpened():
        print(f"Error: Cannot open camera URL: {current_camera_url}")
        return
    
    print(f"Successfully connected to camera: {current_camera_url}")
    
    try:
        while current_camera_url:  # Keep streaming while URL is set
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame")
                break
            
            # Resize for faster processing
            frame = cv2.resize(frame, (640, 480))
            
            # Process with AI
            output = process_frame(frame)
            
            # Encode to JPEG
            _, buffer = cv2.imencode('.jpg', output, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
            
            # Yield frame in multipart format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
    finally:
        cap.release()
        print("Camera released")

@app.route('/')
def index():
    """Main page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/set_camera', methods=['POST'])
def set_camera():
    """Set the camera URL"""
    global current_camera_url
    
    data = request.json
    url = data.get('url', '')
    
    if not url:
        return jsonify({'status': 'error', 'message': 'No URL provided'})
    
    # Test the URL
    test_cap = cv2.VideoCapture(url)
    if not test_cap.isOpened():
        test_cap.release()
        return jsonify({'status': 'error', 'message': 'Cannot connect to camera URL'})
    test_cap.release()
    
    current_camera_url = url
    print(f"Camera URL set to: {url}")
    
    return jsonify({'status': 'success', 'message': 'Camera connected'})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    """Stop the camera"""
    global current_camera_url
    current_camera_url = None
    print("Camera stopped")
    return jsonify({'status': 'success'})

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🔒 Privacy Blur - IP Camera Web App")
    print("="*50)
    print("\n✅ Server starting...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("🎬 Enter your IP camera URL and click 'Start Stream'")
    print("\n⏹️  Press Ctrl+C to stop\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
