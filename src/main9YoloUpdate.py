import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO
import time
import datetime
from torchvision import models, transforms
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support,confusion_matrix
from PIL import Image
from torchvision.models import ResNet50_Weights
import yaml
import cv2
from torchvision.transforms import (
    RandomHorizontalFlip, RandomRotation, ColorJitter, 
    RandomResizedCrop, RandomAffine, RandomPerspective,
    GaussianBlur, RandomAdjustSharpness
)
from torch.utils.data import WeightedRandomSampler

class CustomCNN(nn.Module):
    def __init__(self, num_classes=14):
        super(CustomCNN, self).__init__()
        # Increase dropout rates and add L2 regularization
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),  # Add dropout after each block
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.4),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.5),
        )
        
        # Calculate feature dimensions
        self.feature_dims = self._get_conv_output_dims()
        
        # ResNet feature extractor
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # Freeze early layers
        for param in list(self.resnet.parameters())[:-10]:
            param.requires_grad = False
            
        # Combine features
        self.combine_features = nn.Sequential(
            nn.Linear(self.feature_dims + 1000, 512),  # Reduced from 1024
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, 256),  # Added intermediate layer
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def _get_conv_output_dims(self):
        # Helper function to calculate conv output dimensions
        x = torch.randn(1, 3, 64, 64)  # Assuming 64x64 input images
        x = self.conv_layers(x)
        return x.numel() // x.size(0)

    def forward(self, x):
        # CNN path
        cnn_features = self.conv_layers(x)
        cnn_features = cnn_features.view(cnn_features.size(0), -1)
        
        # ResNet path
        resnet_features = self.resnet(x)
        
        # Combine features
        combined = torch.cat((cnn_features, resnet_features), dim=1)
        return self.combine_features(combined)

class TemporalCNN(nn.Module):
    def __init__(self, base_cnn, num_classes=14, sequence_length=16):
        super(TemporalCNN, self).__init__()
        self.base_cnn = base_cnn
        self.sequence_length = sequence_length
        
        # Add temporal layers
        self.temporal_conv = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Dropout3d(0.5)
        )
        
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.5,
            bidirectional=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        batch_size = x.size(0)
        
        # Process each frame through base CNN
        features = []
        for t in range(self.sequence_length):
            frame_features = self.base_cnn(x[:, t])
            features.append(frame_features)
        
        # Stack features
        features = torch.stack(features, dim=1)
        
        # Apply 3D convolution
        features = self.temporal_conv(features)
        
        # LSTM processing
        lstm_out, _ = self.lstm(features.view(batch_size, self.sequence_length, -1))
        
        # Get final prediction using attention
        attention_weights = torch.softmax(
            self.attention(lstm_out), dim=1
        )
        weighted_features = torch.sum(
            attention_weights.unsqueeze(-1) * lstm_out, dim=1
        )
        
        return self.classifier(weighted_features)

class VideoDataset(Dataset):
    def __init__(self, image_dir, label_dir, sequence_length=16, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.sequence_length = sequence_length
        self.transform = transform
        
        # Thay đổi cách tìm files
        self.image_paths = [
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if f.endswith(('.jpg', '.jpeg', '.png', '.mp4', '.avi'))
        ]

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        if image_path.endswith(('.mp4', '.avi')):  # Nếu là video
            frames = self._load_video(image_path)
        else:  # Nếu là ảnh
            image = Image.open(image_path).convert('RGB')
            frames = [image] * self.sequence_length  # Duplicate ảnh để tạo sequence
        
        label = self._load_label(image_path)
        
        if self.transform:
            frames = [self.transform(f) for f in frames]
        
        return torch.stack(frames), label

    def _load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames evenly
        indices = np.linspace(0, total_frames-1, self.sequence_length, dtype=int)
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frames.append(frame)
        
        cap.release()
        return frames

    def _load_label(self, path):
        # Implement your label loading logic here
        # This is just an example - modify according to your label format
        label_path = os.path.join(
            self.label_dir, 
            os.path.splitext(os.path.basename(path))[0] + '.txt'
        )
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                # Assuming label is the first number in the file
                label = int(float(f.readline().strip().split()[0]))
                return label
        return 0  # Default label if not found

    def __len__(self):
        return len(self.image_paths)
def get_transforms(is_training=True):
    """
    Get transforms for training/validation
    """
    # Base transforms (always applied)
    base_transforms = [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
    
    if is_training:
        # Add augmentation transforms for training
        train_transforms = [
            RandomHorizontalFlip(p=0.5),
            RandomRotation(degrees=10),
            RandomResizedCrop(size=(64, 64), scale=(0.8, 1.0)),
            RandomAffine(degrees=0, translate=(0.1, 0.1)),
            RandomPerspective(distortion_scale=0.2, p=0.5),
            ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        ]
        return transforms.Compose(train_transforms + base_transforms)
    else:
        return transforms.Compose(base_transforms)

def get_sampler(dataset):
    """Create weighted sampler for balanced batches"""
    labels = []
    for _, label in dataset:
        labels.append(label)
    
    class_counts = torch.bincount(torch.tensor(labels))
    weights = 1.0 / class_counts[labels]
    sampler = WeightedRandomSampler(weights, len(weights))
    return sampler

class YOLOCNNModel:
    def __init__(self, data_yaml, weights='yolov8n.pt'):
        # self.test_metrics = None 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.current_epoch = 0
        
        # Initialize YOLO model for inference only
        self.yolo = YOLO(weights)
        
        # Initialize CNN model
        self.cnn = CustomCNN(num_classes=14).to(self.device)

        # Initialize metrics tracking
        self.test_metrics = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'confusion_matrix': None,
            'class_accuracies': {},
            'test_loss': 0.0
        }
        
        # Optimizer for CNN only
        self.optimizer = optim.AdamW(
            self.cnn.parameters(),
            lr=0.001,
            weight_decay=0.01
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.1,
            patience=5,
            verbose=True
        )
        
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.data_yaml = data_yaml
        self.best_accuracy = 0
        self.early_stopping_counter = 0
        self.early_stopping_patience = 10

        # Add class weights calculation
        self.class_weights = self._compute_class_weights(data_yaml)
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights, reduction='none')
        
        # Add weight decay to optimizer
        self.optimizer = optim.AdamW(
            self.cnn.parameters(), 
            lr=0.001,
            weight_decay=0.01  # L2 regularization
        )
        
        # Modify learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,  # Less aggressive reduction
            patience=2,   # Reduced patience
            min_lr=1e-6  # Minimum learning rate
        )

    def _compute_class_weights(self, data_yaml):
        """Compute class weights from training data"""
        class_counts = torch.zeros(14)
        train_dir = "data/dataYOLO2/labels/train"
        
        for label_file in os.listdir(train_dir):
            with open(os.path.join(train_dir, label_file), 'r') as f:
                class_idx = int(float(f.readline().strip().split()[0]))
                class_counts[class_idx] += 1
        
        # Compute inverse frequency weights
        weights = 1.0 / (class_counts + 1)  # Add 1 to avoid division by zero
        weights = weights / weights.sum() * len(weights)  # Normalize
        return weights.to(self.device)

    def train_epoch(self, epoch, total_epochs, train_loader):
        epoch_start_time = time.time()  # Bắt đầu tính thời gian
        self.cnn.train()
        
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass through CNN
            outputs = self.cnn(images)
            
            # Calculate loss
            loss = self.criterion(outputs, targets)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.cnn.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f'Batch [{batch_idx + 1}/{len(train_loader)}] Loss: {loss.item():.4f}')
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        epoch_end_time = time.time()  # Kết thúc tính thời gian
        epoch_duration = epoch_end_time - epoch_start_time  # Tính thời gian đã trôi qua
        
        print(f'\nEpoch [{epoch}/{total_epochs}]')
        print(f'Average Loss: {running_loss/len(train_loader):.4f}')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1-Score: {f1:.4f}')
        print(f'Epoch Duration: {epoch_duration:.2f} seconds')  # In thời gian đã trôi qua
        
        return accuracy
    def validate(self, val_loader):
        """Evaluate model on validation set"""
        self.cnn.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch, targets in val_loader:
                # Giải nén batch (chú ý là lấy ảnh CNN)
                cnn_images, targets = batch[1], targets.to(self.device)
                
                cnn_images = cnn_images.to(self.device)
                outputs = self.cnn(cnn_images)
                _, predicted = outputs.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        print('\nValidation Results:')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1-Score: {f1:.4f}')
        
        return accuracy

    def train(self, train_loader, val_loader, epochs):
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.current_epoch, self.current_epoch + epochs):
            train_loss = self._train_epoch(train_loader)
            val_accuracy, val_loss = self.validate(val_loader)
            
            print(f'\nEpoch [{epoch+1}/{self.current_epoch + epochs}]')
            print(f'Average Train Loss: {train_loss:.4f}')
            print(f'Validation Loss: {val_loss:.4f}')
            print(f'Validation Accuracy: {val_accuracy:.4f}')
            
            # Early stopping based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model()
            else:
                patience_counter += 1
                
            if patience_counter >= self.early_stopping_patience:
                print("Early stopping triggered")
                break
                
            self.scheduler.step(val_accuracy)

    def save_model(self, save_path='model_yolo_cnn.pt'):
        """Save model state"""
        torch.save({
            'epoch': self.current_epoch,
            'cnn_state_dict': self.cnn.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_accuracy': self.best_accuracy,
            'test_metrics': self.test_metrics,
        }, save_path)
        print(f"Model saved to {save_path} with test metrics")

    def resume_training(self, checkpoint_path='model_yolo_cnn.pt'):
        """Resume training from checkpoint"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.cnn.load_state_dict(checkpoint['cnn_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.best_accuracy = checkpoint['best_accuracy']
            start_epoch = checkpoint.get('epoch', 0)
            print(f"Resumed from epoch {start_epoch} with accuracy {self.best_accuracy:.4f}")
            return start_epoch
        return 0

    def test(self, test_loader):
        """
        Evaluate model on test set and compute comprehensive metrics
        """
        print("\nEvaluating on test set...")
        self.cnn.eval()
        
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        class_correct = {}
        class_total = {}
        
        # Initialize counters for each class
        num_classes = 14  # Based on your data.yaml
        for i in range(num_classes):
            class_correct[i] = 0
            class_total[i] = 0
        
        with torch.no_grad():
            for batch_idx, (batch, targets) in enumerate(test_loader):
                # Giải nén batch (chú ý là lấy ảnh CNN)
                cnn_images, targets = batch[1], targets.to(self.device)
                
                cnn_images = cnn_images.to(self.device)
                
                # Forward pass
                outputs = self.cnn(cnn_images)
                loss = self.criterion(outputs, targets).mean()
                total_loss += loss.item()
                
                # Get predictions
                _, predicted = outputs.max(1)
                
                # Update metrics
                for i in range(len(targets)):
                    label = targets[i].item()
                    class_total[label] = class_total.get(label, 0) + 1
                    if label == predicted[i].item():
                        class_correct[label] = class_correct.get(label, 0) + 1
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())
                
                # Print progress
                if (batch_idx + 1) % 10 == 0:
                    print(f'Processed {batch_idx + 1}/{len(test_loader)} batches')
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        # Calculate per-class accuracy
        class_accuracies = {}
        for i in range(num_classes):
            if class_total[i] > 0:
                class_accuracies[i] = class_correct[i] / class_total[i]
            else:
                class_accuracies[i] = 0.0
        
        # Update test_metrics dictionary
        self.test_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': confusion_matrix(all_labels, all_predictions),
            'class_accuracies': class_accuracies,
            'test_loss': total_loss / len(test_loader)
        }
        
        # Print detailed results
        print('\nTest Results:')
        print(f'Overall Accuracy: {accuracy:.4f}')
        print(f'Overall Precision: {precision:.4f}')
        print(f'Overall Recall: {recall:.4f}')
        print(f'Overall F1-Score: {f1:.4f}')
        print(f'Average Test Loss: {self.test_metrics["test_loss"]:.4f}')
        print('\nPer-class Accuracy:')
        
        # Load class names from data.yaml
        with open(self.data_yaml, 'r') as f:
            data_dict = yaml.safe_load(f)
            class_names = data_dict.get('names', {})
        
        for class_idx, acc in class_accuracies.items():
            class_name = class_names.get(class_idx, f'Class {class_idx}')
            print(f'{class_name}: {acc:.4f}')
        
        return self.test_metrics
    def load_test_metrics(self, checkpoint_path='model_yolo_cnn.pt'):
        """Load test metrics from saved checkpoint"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.test_metrics = checkpoint.get('test_metrics', None)
            if self.test_metrics:
                print("Loaded test metrics from checkpoint:")
                print(f"Accuracy: {self.test_metrics['accuracy']:.4f}")
                print(f"Precision: {self.test_metrics['precision']:.4f}")
                print(f"Recall: {self.test_metrics['recall']:.4f}")
                print(f"F1-Score: {self.test_metrics['f1']:.4f}")
            else:
                print("No test metrics found in checkpoint")
        else:
            print(f"Checkpoint file not found: {checkpoint_path}")

def main():
    # Initialize datasets with video processing
    # train_dataset = VideoDataset(
    #     video_dir="data/videos/train",
    #     label_dir="data/labels/train",
    #     transform=get_transforms(is_training=True)
    # )
    
    # Get transforms for validation/test data (no augmentation)
    # val_yolo_transform, val_cnn_transform = get_transforms(is_training=False)
    
    # Load datasets with appropriate transforms
    train_dataset = VideoDataset(
        image_dir="data/dataYOLO2/images/train",
        label_dir="data/dataYOLO2/labels/train",
        transform=get_transforms(is_training=True)  # Chỉ lấy một transform
    )
    
    val_dataset = VideoDataset(
        image_dir="data/dataYOLO2/images/val",
        label_dir="data/dataYOLO2/labels/val",
        transform=get_transforms(is_training=False)  # Chỉ lấy một transform
    )
    
    test_dataset = VideoDataset(
        image_dir="data/dataYOLO2/images/test",
        label_dir="data/dataYOLO2/labels/test",
        transform=get_transforms(is_training=False)  # Chỉ lấy một transform
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler=get_sampler(train_dataset),
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )
    
    # Initialize model
    data_yaml = "data/dataYOLO2/data.yaml"
    model = YOLOCNNModel(data_yaml)
    
    # Resume training if checkpoint exists
    start_epoch = model.resume_training()
    
    # Train model
    remaining_epochs = 5 - start_epoch
    if remaining_epochs > 0:
        print(f"Starting training from epoch {start_epoch + 1}")
        model.train(train_loader, val_loader, epochs=remaining_epochs)
    else:
        print("Training already completed")

    # Evaluate on test set
    print("Evaluating model on test set...")
    test_metrics = model.test(test_loader)
    
    # Save final model with test metrics
    model.save_model()

if __name__ == "__main__":
    main()