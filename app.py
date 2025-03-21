from flask import Flask, request, jsonify
import urllib.request
import numpy as np
import cv2
import torch
from ultralytics import YOLO

app = Flask(__name__)

# ESP32-CAM Stream URL (Update with your ESP32 IP)
ESP32_URL = "http://192.168.205.210/capture"

# Load YOLOv8 model (Ensure you use a lightweight model for better speed on Render)
model = YOLO("yolov8n.pt")

def get_position(x, frame_width):
    if x < frame_width // 3:
        return "Left"
    elif x > 2 * frame_width // 3:
        return "Right"
    else:
        return "Center"

def get_proximity(y1, y2, frame_height):
    box_height = y2 - y1
    return box_height / frame_height  # Normalize proximity value

@app.route("/detect", methods=["GET"])
def detect_objects():
    try:
        # Fetch image from ESP32-CAM
        img_resp = urllib.request.urlopen(ESP32_URL)
        img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_np, -1)
        frame_height, frame_width, _ = frame.shape
        
        # Perform object detection
        results = model(frame)
        
        detected_objects = []
        for detection in results[0].boxes.data:
            x1, y1, x2, y2 = map(int, detection[:4])
            obj_class = int(detection[5])  # Object class index
            object_name = model.names[obj_class]
            
            # Calculate object position & proximity
            object_x_center = (x1 + x2) // 2
            position = get_position(object_x_center, frame_width)
            proximity = get_proximity(y1, y2, frame_height)
            
            detected_objects.append({
                "name": object_name,
                "position": position,
                "proximity": proximity
            })
        
        # Sort objects by proximity (closest first)
        detected_objects.sort(key=lambda obj: obj["proximity"], reverse=True)
        
        return jsonify(detected_objects)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
