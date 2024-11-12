import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from main9YoloUpdate import TemporalCNN, CustomCNN

class VideoBehaviorPredictor:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.transform = self._get_transform()
        self.behaviors = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 
                         'Explosion', 'Fighting', 'RoadAccidents', 'Robbery', 
                         'Shooting', 'Shoplifting', 'Stealing', 'Vandalism', 
                         'NormalVideos']

    def _load_model(self, model_path):
        base_cnn = CustomCNN(num_classes=14)
        model = TemporalCNN(base_cnn, num_classes=14)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model

    def _get_transform(self):
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict_video(self, video_path):
        # Extract frames
        frames = self._extract_frames(video_path)
        if len(frames) == 0:
            return "Error: Could not process video"

        # Process frames in batches
        predictions = []
        with torch.no_grad():
            for i in range(0, len(frames), 16):
                batch_frames = frames[i:i+16]
                if len(batch_frames) < 16:
                    # Pad if necessary
                    batch_frames.extend([batch_frames[-1]] * (16 - len(batch_frames)))
                
                # Transform and predict
                batch_tensor = torch.stack([
                    self.transform(Image.fromarray(f)) for f in batch_frames
                ]).unsqueeze(0).to(self.device)
                
                outputs = self.model(batch_tensor)
                pred = torch.softmax(outputs, dim=1)
                predictions.append(pred.cpu().numpy())

        # Aggregate predictions
        final_pred = np.mean(predictions, axis=0)
        behavior_idx = np.argmax(final_pred)
        confidence = final_pred[0][behavior_idx]

        return {
            'behavior': self.behaviors[behavior_idx],
            'confidence': float(confidence),
            'predictions': {
                self.behaviors[i]: float(final_pred[0][i])
                for i in range(len(self.behaviors))
            }
        }

    def _extract_frames(self, video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            
        cap.release()
        return frames

# Example usage
if __name__ == "__main__":
    predictor = VideoBehaviorPredictor('model_yolo_cnn.pt')
    result = predictor.predict_video('path/to/video.mp4')
    print(f"Detected behavior: {result['behavior']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print("\nAll predictions:")
    for behavior, conf in result['predictions'].items():
        print(f"{behavior}: {conf:.2f}")