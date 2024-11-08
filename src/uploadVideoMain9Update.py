# app.py
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from ultralytics import YOLO
import threading
import time
import json
import logging
import yaml
from PIL import Image
from pathlib import Path
from model import CustomCNN  # Your custom CNN model

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'static/results'
checkpoint_path = 'model_yolo_cnn.pt'
data_yaml = 'data/processed/dataYOLO2/data.yaml'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Global processing status
processing_status = {}

# Load behavior detection model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
behavior_model = CustomCNN(num_classes=14).to(device)
checkpoint = torch.load(checkpoint_path, map_location=device)
behavior_model.load_state_dict(checkpoint['cnn_state_dict'])
behavior_model.eval()

# Load class names for behavior
with open(data_yaml, 'r') as f:
    data_dict = yaml.safe_load(f)
    behavior_class_names = data_dict.get('names', {})

# Transform for behavior detection
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_frame_behavior(frame, model, transform, device):
    """Process a single frame for behavior detection"""
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        input_tensor = transform(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = output.max(1)
            behavior = behavior_class_names[predicted.item()]
            
        return behavior
    except Exception as e:
        logger.error(f"Error processing frame for behavior: {str(e)}")
        raise

class CombinedVideoProcessor:
    def __init__(self):
        self.yolo_model = YOLO('yolov8n.pt')

    def process_video(self, video_path, output_path, task_id):
        try:
            processing_status[task_id] = {
                'status': 'processing',
                'progress': 0,
                'results': []
            }

            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (width, height)
            )

            frame_count = 0
            results_data = []

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                # YOLO object detection
                yolo_results = self.yolo_model(frame)
                annotated_frame = yolo_results[0].plot()

                # Behavior detection
                behavior = process_frame_behavior(frame, behavior_model, transform, device)

                # Add behavior text to frame
                cv2.putText(
                    annotated_frame,
                    f"Behavior: {behavior}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

                writer.write(annotated_frame)

                # Store detection results
                frame_results = {
                    'frame': frame_count,
                    'behavior': behavior,
                    'objects': []
                }

                for r in yolo_results:
                    for box in r.boxes:
                        frame_results['objects'].append({
                            'class': r.names[int(box.cls)],
                            'confidence': float(box.conf),
                            'bbox': box.xyxy[0].tolist()
                        })

                results_data.append(frame_results)

                frame_count += 1
                progress = int((frame_count / total_frames) * 100)
                processing_status[task_id]['progress'] = progress

            cap.release()
            writer.release()

            # Save results to JSON
            results_json_path = os.path.join(
                app.config['RESULTS_FOLDER'],
                f'{task_id}_results.json'
            )
            with open(results_json_path, 'w') as f:
                json.dump(results_data, f)

            processing_status[task_id] = {
                'status': 'completed',
                'progress': 100,
                'results': results_data
            }

        except Exception as e:
            processing_status[task_id] = {
                'status': 'error',
                'error': str(e)
            }
            logger.error(f"Error processing video: {str(e)}")

video_processor = CombinedVideoProcessor()

@app.route('/')
def index():
    return render_template('index9update.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        task_id = f"{int(time.time())}_{filename}"
        
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)
        
        output_filename = f'processed_{filename}'
        output_path = os.path.join(app.config['RESULTS_FOLDER'], output_filename)
        
        thread = threading.Thread(
            target=video_processor.process_video,
            args=(video_path, output_path, task_id)
        )
        thread.start()
        
        return jsonify({
            'task_id': task_id,
            'message': 'Processing started'
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/status/<task_id>')
def get_status(task_id):
    if task_id in processing_status:
        return jsonify(processing_status[task_id])
    return jsonify({'error': 'Task not found'}), 404

@app.route('/results/<path:filename>')
def get_result(filename):
    return send_from_directory(
        app.config['RESULTS_FOLDER'],
        filename,
        mimetype='video/mp4'
    )

if __name__ == '__main__':
    app.run(debug=True)