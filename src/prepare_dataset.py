import os
import shutil
import yaml
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import io

BEHAVIORS = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 
            'Fighting', 'RoadAccidents', 'Robbery', 'Shooting', 'Shoplifting', 
            'Stealing', 'Vandalism', 'NormalVideos']

# Thiết lập mã hóa cho stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def prepare_yolo_dataset(input_dir, output_dir):
    """Chuyển đổi dataset sang format YOLO với train/val/test splits"""
    # Tạo cấu trúc thư mục YOLO
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)
    
    # Tạo file data.yaml
    data = {
        'path': output_dir,
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {i: name for i, name in enumerate(BEHAVIORS)},
        'nc': len(BEHAVIORS)  # number of classes
    }
    
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        yaml.dump(data, f)
    
    # Chuyển đổi ảnh và tạo labels
    for idx, behavior in enumerate(BEHAVIORS):
        behavior_dir = os.path.join(input_dir, behavior)
        if not os.path.exists(behavior_dir):
            continue
            
        images = [f for f in os.listdir(behavior_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Chia thành train/val/test với tỷ lệ 70/20/10
        train_images, temp_images = train_test_split(images, train_size=0.7, random_state=42)
        val_images, test_images = train_test_split(temp_images, train_size=0.67, random_state=42)
        
        # Hàm helper để xử lý mỗi split
        def process_split(image_list, split_name):
            for img in image_list:
                # Copy ảnh
                src = os.path.join(behavior_dir, img)
                dst = os.path.join(output_dir, 'images', split_name, f'{behavior}_{img}')
                shutil.copy2(src, dst)
                
                # Tạo label file
                img_array = np.array(Image.open(src))
                h, w = img_array.shape[:2]
                
                # Format YOLO: <class> <x_center> <y_center> <width> <height>
                label_content = f"{idx} 0.5 0.5 1.0 1.0"
                
                label_path = os.path.join(output_dir, 'labels', split_name,
                                        f'{behavior}_{os.path.splitext(img)[0]}.txt')
                with open(label_path, 'w') as f:
                    f.write(label_content)
        
        # Xử lý cho từng split
        process_split(train_images, 'train')
        process_split(val_images, 'val')
        process_split(test_images, 'test')
        
        # In thông tin về số lượng ảnh trong mỗi split
        print(f"\nBehavior: {behavior}")
        print(f"Total images: {len(images)}")
        print(f"Train images: {len(train_images)}")
        print(f"Validation images: {len(val_images)}")
        print(f"Test images: {len(test_images)}")

def verify_dataset(output_dir):
    """Kiểm tra tính toàn vẹn của dataset sau khi chuyển đổi"""
    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(output_dir, 'images', split)
        label_dir = os.path.join(output_dir, 'labels', split)
        
        images = set(os.path.splitext(f)[0] for f in os.listdir(img_dir))
        labels = set(os.path.splitext(f)[0] for f in os.listdir(label_dir))
        
        if images != labels:
            print(f"\nWarning: Mismatch in {split} set!")
            print(f"Images without labels: {images - labels}")
            print(f"Labels without images: {labels - images}")
        else:
            print(f"\n{split} set: OK")
            print(f"Number of samples: {len(images)}")

def main():
    input_dir = "data/raw/Train"
    output_dir = "data/processed/dataYOLO2"
    
    print("Bắt đầu chuyển đổi dataset...")
    prepare_yolo_dataset(input_dir, output_dir)
    
    print("\nKiểm tra tính toàn vẹn của dataset...")
    verify_dataset(output_dir)
    
    print("\nĐã hoàn thành chuyển đổi dataset!")

if __name__ == "__main__":
    main()
