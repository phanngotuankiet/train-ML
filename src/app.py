import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from predict_video import VideoBehaviorPredictor
import cv2
import time

app = Flask(__name__)

# Cấu hình
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

# Đảm bảo thư mục tồn tại
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Lưu file upload
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        filename_with_timestamp = f"{timestamp}_{filename}"
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename_with_timestamp)
        file.save(upload_path)
        
        try:
            # Khởi tạo predictor
            predictor = VideoBehaviorPredictor('models/model_yolo_cnn.pt')
            
            # Phân tích video
            result = predictor.predict_video(upload_path)
            
            # Tạo video kết quả với nhãn
            result_filename = f"result_{filename_with_timestamp}"
            result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
            
            # Đọc video gốc và thêm nhãn
            cap = cv2.VideoCapture(upload_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(result_path, fourcc, fps, (width, height))
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Thêm text vào frame
                text = f"Behavior: {result['behavior']} ({result['confidence']:.2f})"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 255, 0), 2)
                
                out.write(frame)
            
            cap.release()
            out.release()
            
            return jsonify({
                'success': True,
                'result': result,
                'result_video': f'/static/results/{result_filename}'
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)