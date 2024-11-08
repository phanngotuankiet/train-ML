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

# class YourDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         """
#         Custom Dataset for loading images and their labels.
        
#         Args:
#             root_dir (str): Directory with all the images organized in class folders
#             transform (callable, optional): Optional transform to be applied on images
#         """
#         self.root_dir = root_dir
#         self.transform = transform
#         self.classes = sorted(os.listdir(root_dir))  # Get class names from folder names
#         self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
#         # Create list of (image_path, class_idx) tuples
#         self.samples = []
#         for class_name in self.classes:
#             class_dir = os.path.join(root_dir, class_name)
#             if not os.path.isdir(class_dir):
#                 continue
            
#             class_idx = self.class_to_idx[class_name]
#             for img_name in os.listdir(class_dir):
#                 if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
#                     img_path = os.path.join(class_dir, img_name)
#                     self.samples.append((img_path, class_idx))
    
#     def __len__(self):
#         """Returns the total number of samples in the dataset"""
#         return len(self.samples)
    
#     def __getitem__(self, idx):
#         """
#         Returns one sample of data, data and label (image and class).
        
#         Args:
#             idx (int): Index of the sample to fetch
            
#         Returns:
#             tuple: (image, label) where label is the class index
#         """
#         img_path, label = self.samples[idx]
        
#         try:
#             # Load image using PIL
#             image = Image.open(img_path).convert('RGB')
            
#             # Apply transformations if any
#             if self.transform:
#                 image = self.transform(image)
            
#             return image, label
            
#         except Exception as e:
#             print(f"Error loading image {img_path}: {str(e)}")
#             # Return a black image and the label in case of error
#             if self.transform:
#                 dummy_img = self.transform(Image.new('RGB', (64, 64), color='black'))
#             else:
#                 dummy_img = torch.zeros((3, 64, 64))
#             return dummy_img, label
    
#     def get_class_names(self):
#         """Returns list of class names"""
#         return self.classes
    
#     def get_num_classes(self):
#         """Returns total number of classes"""
#         return len(self.classes)
class YourDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        
        # List all image files
        self.image_paths = [
            os.path.join(image_dir, img_name) 
            for img_name in os.listdir(image_dir) 
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Get corresponding label file
        label_path = os.path.join(
            self.label_dir, 
            os.path.splitext(os.path.basename(img_path))[0] + '.txt'
        )
        
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            
            # Load and process label
            with open(label_path, 'r') as f:
                # YOLO format: class x_center y_center width height
                # We only need the class (first number)
                line = f.readline().strip().split()[0]  # Get first number only
                label = int(float(line))  # Convert to int, handling float strings
            
            return image, label

        except Exception as e:
            print(f"Error loading data for {img_path}: {str(e)}")
            # Return dummy data in case of error
            dummy_img = torch.zeros((3, 64, 64)) if not self.transform else \
                       self.transform(Image.new('RGB', (64, 64), color='black'))
            return dummy_img, 0  # Return 0 as default class

# class YOLOCNNModel:
#     def __init__(self, data_yaml, weights='yolov8n.pt'):
#         """
#         Initialize YOLOCNNModel
#         Args:
#             data_yaml (str): Path to data.yaml file
#             weights (str): Path to YOLO weights file
#         """
#         self.test_metrics = None 
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.current_epoch = 0
        
#         # Initialize YOLO model
#         self.yolo = YOLO(weights)
        
#         # Initialize CNN model with 14 classes (fixed number for your dataset)
#         self.cnn = CustomCNN(num_classes=14).to(self.device)
        
#         # Optimizer with learning rate scheduler
#         self.optimizer = optim.AdamW(
#             self.cnn.parameters(), 
#             lr=0.001, 
#             weight_decay=0.01
#         )
        
#         # Learning rate scheduler
#         self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#             self.optimizer, 
#             mode='max', 
#             factor=0.1, 
#             patience=5,
#             verbose=True
#         )
        
#         # Loss function with class weights
#         self.criterion = nn.CrossEntropyLoss(reduction='none')
        
#         self.data_yaml = data_yaml
#         self.best_accuracy = 0
#         self.early_stopping_counter = 0
#         self.early_stopping_patience = 10
        
#     def calculate_class_weights(self, labels):
#         """Tính toán class weights để xử lý imbalanced data"""
#         class_counts = torch.bincount(labels)
#         total = len(labels)
#         weights = total / (len(class_counts) * class_counts.float())
#         return weights.to(self.device)
        
#     def train_epoch(self, epoch, total_epochs, train_loader):
#         epoch_start_time = time.time()
        
#         # Train YOLO
#         results = self.yolo.train(
#             data=self.data_yaml,
#             epochs=1,
#             imgsz=64,
#             batch=32,
#             exist_ok=True,
#             resume=True
#         )
        
#         # Train CNN
#         self.cnn.train()
#         running_loss = 0.0
#         all_predictions = []
#         all_labels = []
        
#         for batch_idx, (images, targets) in enumerate(train_loader):
#             images, targets = images.to(self.device), targets.to(self.device)
            
#             # Get YOLO features
#             with torch.no_grad():
#                 yolo_features = self.yolo.predict(images, verbose=False)
            
#             # Forward pass
#             self.optimizer.zero_grad()
#             outputs = self.cnn(images)  # Using original images for CNN
            
#             # Calculate loss with class weights
#             class_weights = self.calculate_class_weights(targets)
#             loss = self.criterion(outputs, targets)
#             weighted_loss = (loss * class_weights[targets]).mean()
            
#             # Backward pass
#             weighted_loss.backward()
            
#             # Gradient clipping
#             torch.nn.utils.clip_grad_norm_(self.cnn.parameters(), max_norm=1.0)
            
#             self.optimizer.step()
            
#             running_loss += weighted_loss.item()
#             _, predicted = outputs.max(1)
            
#             all_predictions.extend(predicted.cpu().numpy())
#             all_labels.extend(targets.cpu().numpy())
            
#             # Print batch progress
#             if (batch_idx + 1) % 10 == 0:
#                 print(f'Batch [{batch_idx + 1}/{len(train_loader)}] Loss: {weighted_loss.item():.4f}')
        
#         # Calculate metrics
#         accuracy = accuracy_score(all_labels, all_predictions)
#         precision, recall, f1, _ = precision_recall_fscore_support(
#             all_labels, 
#             all_predictions, 
#             average='weighted'
#         )
        
#         # Calculate epoch time
#         epoch_time = time.time() - epoch_start_time
        
#         # Print epoch results
#         print(f'\nEpoch [{epoch}/{total_epochs}]')
#         print(f'Average Loss: {running_loss/len(train_loader):.4f}')
#         print(f'Accuracy: {accuracy:.4f}')
#         print(f'Precision: {precision:.4f}')
#         print(f'Recall: {recall:.4f}')
#         print(f'F1-Score: {f1:.4f}')
#         print(f'Epoch Time: {datetime.timedelta(seconds=int(epoch_time))}')
        
#         return accuracy

#     def validate(self, val_loader):
#         """Đánh giá model trên validation set"""
#         self.cnn.eval()
#         all_predictions = []
#         all_labels = []
        
#         with torch.no_grad():
#             for images, targets in val_loader:
#                 images, targets = images.to(self.device), targets.to(self.device)
#                 outputs = self.cnn(images)
#                 _, predicted = outputs.max(1)
                
#                 all_predictions.extend(predicted.cpu().numpy())
#                 all_labels.extend(targets.cpu().numpy())
        
#         accuracy = accuracy_score(all_labels, all_predictions)
#         precision, recall, f1, _ = precision_recall_fscore_support(
#             all_labels, 
#             all_predictions, 
#             average='weighted'
#         )
        
#         print('\nValidation Results:')
#         print(f'Accuracy: {accuracy:.4f}')
#         print(f'Precision: {precision:.4f}')
#         print(f'Recall: {recall:.4f}')
#         print(f'F1-Score: {f1:.4f}')
        
#         return accuracy
class EnhancedDataset(Dataset):
    def __init__(self, image_dir, label_dir, yolo_transform=None, cnn_transform=None):
        """
        Args:
            image_dir: Directory containing images
            label_dir: Directory containing labels
            yolo_transform: Transform for YOLO model
            cnn_transform: Transform for CNN model
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.yolo_transform = yolo_transform
        self.cnn_transform = cnn_transform
        
        # List all image files
        self.image_paths = [
            os.path.join(image_dir, img_name) 
            for img_name in os.listdir(image_dir) 
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

    def __getitem__(self, idx):
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            # Apply transforms with error handling
            try:
                yolo_image = self.yolo_transform(image) if self.yolo_transform else transforms.ToTensor()(image)
                cnn_image = self.cnn_transform(image) if self.cnn_transform else transforms.ToTensor()(image)
            except Exception as e:
                print(f"Transform error for {img_path}: {str(e)}")
                # Fallback to basic transform
                basic_transform = transforms.Compose([
                    transforms.Resize((64, 64)),
                    transforms.ToTensor()
                ])
                yolo_image = basic_transform(image)
                cnn_image = basic_transform(image)
            
            return (yolo_image, cnn_image), label
            
        except Exception as e:
            print(f"Error loading data for {img_path}: {str(e)}")
            # Return dummy data in case of error
            dummy = torch.zeros((3, 64, 64))
            return (dummy, dummy), 0

    def __len__(self):
        return len(self.image_paths)

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
        train_dir = "data/processed/dataYOLO2/labels/train"
        
        for label_file in os.listdir(train_dir):
            with open(os.path.join(train_dir, label_file), 'r') as f:
                class_idx = int(float(f.readline().strip().split()[0]))
                class_counts[class_idx] += 1
        
        # Compute inverse frequency weights
        weights = 1.0 / (class_counts + 1)  # Add 1 to avoid division by zero
        weights = weights / weights.sum() * len(weights)  # Normalize
        return weights.to(self.device)

    def train_epoch(self, epoch, total_epochs, train_loader):
            epoch_start_time = time.time()
            self.cnn.train()
            
            running_loss = 0.0
            all_predictions = []
            all_labels = []
            
            for batch_idx, ((yolo_images, cnn_images), targets) in enumerate(train_loader):
                yolo_images = yolo_images.to(self.device)
                cnn_images = cnn_images.to(self.device)
                targets = targets.to(self.device)
                
                # Get YOLO features using properly normalized images
                with torch.no_grad():
                    yolo_results = self.yolo.predict(yolo_images, verbose=False)
                
                # Forward pass through CNN using properly normalized images
                outputs = self.cnn(cnn_images)
                
                # Rest of the training code remains same
                class_weights = self.calculate_class_weights(targets)
                loss = self.criterion(outputs, targets)
                weighted_loss = (loss * class_weights[targets]).mean()
                
                self.optimizer.zero_grad()
                weighted_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.cnn.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                running_loss += weighted_loss.item()
                _, predicted = outputs.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())
                
                if (batch_idx + 1) % 10 == 0:
                    print(f'Batch [{batch_idx + 1}/{len(train_loader)}] Loss: {weighted_loss.item():.4f}')
            
            # Calculate metrics
            accuracy = accuracy_score(all_labels, all_predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_predictions, average='weighted'
            )
            
            print(f'\nEpoch [{epoch}/{total_epochs}]')
            print(f'Average Loss: {running_loss/len(train_loader):.4f}')
            print(f'Accuracy: {accuracy:.4f}')
            print(f'Precision: {precision:.4f}')
            print(f'Recall: {recall:.4f}')
            print(f'F1-Score: {f1:.4f}')
            
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


   


def get_transforms(is_training=True):
    """
    Get transforms for both YOLO and CNN paths
    Args:
        is_training (bool): Whether to include augmentation transforms
    """
    # Base transforms (always applied)
    base_transforms = [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ]
    
    # Augmentation transforms (only during training)
    aug_transforms = [
        # Geometric transforms
        RandomHorizontalFlip(p=0.5),
        RandomRotation(degrees=10),
        RandomResizedCrop(size=(64, 64), scale=(0.8, 1.0)),
        RandomAffine(degrees=0, translate=(0.1, 0.1)),
        RandomPerspective(distortion_scale=0.2, p=0.5),
        
        # Color/intensity transforms
        ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        RandomAdjustSharpness(sharpness_factor=2, p=0.5),
    ]
    
    # CNN-specific normalization
    cnn_normalize = [
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
    
    if is_training:
        yolo_transform = transforms.Compose(aug_transforms + base_transforms)
        cnn_transform = transforms.Compose(aug_transforms + base_transforms + cnn_normalize)
    else:
        yolo_transform = transforms.Compose(base_transforms)
        cnn_transform = transforms.Compose(base_transforms + cnn_normalize)
    
    return yolo_transform, cnn_transform

def get_sampler(dataset):
    """Create weighted sampler for balanced batches"""
    labels = []
    for _, label in dataset:
        labels.append(label)
    
    class_counts = torch.bincount(torch.tensor(labels))
    weights = 1.0 / class_counts[labels]
    sampler = WeightedRandomSampler(weights, len(weights))
    return sampler

def main():
    # Get transforms for training data
    train_yolo_transform, train_cnn_transform = get_transforms(is_training=True)
    
    # Get transforms for validation/test data (no augmentation)
    val_yolo_transform, val_cnn_transform = get_transforms(is_training=False)
    
    # Load datasets with appropriate transforms
    train_dataset = EnhancedDataset(
        image_dir="data/processed/dataYOLO2/images/train",
        label_dir="data/processed/dataYOLO2/labels/train",
        yolo_transform=train_yolo_transform,
        cnn_transform=train_cnn_transform
    )
    
    val_dataset = EnhancedDataset(
        image_dir="data/processed/dataYOLO2/images/val",
        label_dir="data/processed/dataYOLO2/labels/val",
        yolo_transform=val_yolo_transform,
        cnn_transform=val_cnn_transform
    )
    
    test_dataset = EnhancedDataset(
        image_dir="data/processed/dataYOLO2/images/test",
        label_dir="data/processed/dataYOLO2/labels/test",
        yolo_transform=val_yolo_transform,
        cnn_transform=val_cnn_transform
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler=get_sampler(train_dataset),  # Add weighted sampler
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
    data_yaml = "data/processed/dataYOLO2/data.yaml"
    model = YOLOCNNModel(data_yaml)  # Just pass data_yaml path
    
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