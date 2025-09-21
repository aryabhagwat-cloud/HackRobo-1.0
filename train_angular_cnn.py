#!/usr/bin/env python3
"""
Optimized CNN Training Script for Angular View Disease Detection
Specifically designed for crawler camera angles - completes in under 4 hours
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
from PIL import Image
import numpy as np
from pathlib import Path
import json
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import time


class AngularPlantDiseaseDataset(Dataset):
    """Dataset optimized for angular view augmentation"""
    
    def __init__(self, image_paths, labels, transform=None, use_albumentations=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.use_albumentations = use_albumentations
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        if self.use_albumentations:
            # Load with OpenCV for Albumentations
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']
        else:
            # Use PIL for torchvision
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        
        return image, label


def load_datasets_optimized():
    """Load datasets with optimized settings for speed"""
    print("Loading Datasets (Optimized for Speed)...")
    print("=" * 60)
    
    all_images = []
    all_labels = []
    class_mapping = {}
    
    # Dataset configuration
    datasets = {
        'potato': {
            'path': r"C:\Users\yashv.HPLAPTOP\OneDrive\Documents\data)vijay\potato_split\kaggle\working\data\potato",
            'max_images': 200,  # Reduced for speed
            'subdir': None
        },
        'tomato': {
            'path': r"C:\Users\yashv.HPLAPTOP\OneDrive\Documents\data)vijay\tomato_split\tomato_split\train",
            'max_images': 300,  # Reduced for speed
            'subdir': None
        },
        'pepper': {
            'path': r"C:\Users\yashv.HPLAPTOP\OneDrive\Documents\data)vijay\pepper_split\pepper_split\train",
            'max_images': 150,  # Reduced for speed
            'subdir': None
        }
    }
    
    class_idx = 0
    
    for crop_name, config in datasets.items():
        print(f"\n📁 Loading {crop_name.upper()} dataset...")
        dataset_path = Path(config['path'])
        
        if not dataset_path.exists():
            print(f"Dataset not found: {dataset_path}")
            continue
        
        # Get class directories
        class_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
        crop_classes = [d.name for d in class_dirs]
        print(f"Found {len(crop_classes)} classes: {crop_classes}")
        
        for class_name in crop_classes:
            class_dir = dataset_path / class_name
            print(f"  Loading {class_name}...")
            
            # Get image files
            image_files = []
            for ext in ['*.JPG', '*.jpg', '*.jpeg', '*.JPEG', '*.png', '*.PNG']:
                image_files.extend(list(class_dir.glob(ext)))
            
            # Limit images for speed
            max_images = config.get('max_images', 200)
            if len(image_files) > max_images:
                image_files = random.sample(image_files, max_images)
            
            print(f"    Using {len(image_files)} images")
            
            # Add to dataset
            for img_path in image_files:
                all_images.append(str(img_path))
                all_labels.append(class_idx)
            
            class_mapping[class_name] = class_idx
            class_idx += 1
    
    # Create reverse mapping
    idx_to_class = {v: k for k, v in class_mapping.items()}
    
    print(f"\nTotal dataset loaded:")
    print(f"   Images: {len(all_images)}")
    print(f"   Classes: {len(class_mapping)}")
    print(f"   Class mapping: {class_mapping}")
    
    return all_images, all_labels, class_mapping, idx_to_class


def create_angular_data_loaders(image_paths, labels, batch_size=16, train_split=0.8):
    """Create data loaders with angular view augmentation"""
    print(f"\nCreating Angular View Data Loaders...")
    
    # Split data
    n_total = len(image_paths)
    n_train = int(n_total * train_split)
    
    # Shuffle indices
    indices = list(range(n_total))
    random.shuffle(indices)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    # Split data
    train_images = [image_paths[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_images = [image_paths[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]
    
    print(f"   Train: {len(train_images)} images")
    print(f"   Validation: {len(val_images)} images")
    
    # Angular view augmentation (optimized for crawler)
    train_transform = A.Compose([
        A.Resize(224, 224),
        # Angular view simulation
        A.Perspective(scale=(0.1, 0.3), p=0.8),  # Simulate angular views
        A.Rotate(limit=45, p=0.9),               # Camera angle variations
        A.RandomRotate90(p=0.5),                 # Different orientations
        A.HorizontalFlip(p=0.5),                 # Left/right views
        A.VerticalFlip(p=0.3),                   # Upside down views
        # Lighting and quality variations
        A.RandomBrightnessContrast(
            brightness_limit=0.3, 
            contrast_limit=0.3, 
            p=0.8
        ),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.5),
        ], p=0.3),
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        ], p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Create datasets
    train_dataset = AngularPlantDiseaseDataset(train_images, train_labels, train_transform, use_albumentations=True)
    val_dataset = AngularPlantDiseaseDataset(val_images, val_labels, val_transform, use_albumentations=True)
    
    # Create data loaders with reduced workers for stability
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    
    return train_loader, val_loader


class AngularPlantDiseaseCNN(nn.Module):
    """MobileNetV2 optimized for angular view disease detection"""
    
    def __init__(self, num_classes):
        super(AngularPlantDiseaseCNN, self).__init__()
        
        # Load pretrained MobileNetV2
        self.backbone = mobilenet_v2(pretrained=True)
        
        # Replace classifier with dropout for better generalization
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.backbone.last_channel, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


def train_angular_model(model, train_loader, val_loader, num_epochs=15, learning_rate=0.001):
    """Train model with angular view optimization"""
    print(f"\nTraining Angular View CNN Model...")
    print("=" * 60)
    
    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model.to(device)
    
    # Optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training history
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0.0
    best_model_state = None
    patience = 3  # Reduced patience for speed
    patience_counter = 0
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (images, labels) in enumerate(train_pbar):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation")
            for images, labels in val_pbar:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Update learning rate
        scheduler.step()
        
        # Early stopping and best model saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Store history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Best Val Acc: {best_val_acc:.2f}%")
        print(f"Epoch Time: {epoch_time:.1f}s, Total Time: {total_time/60:.1f}m")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    total_training_time = time.time() - start_time
    print(f"\nTotal Training Time: {total_training_time/60:.1f} minutes")
    
    return model, train_losses, val_losses, val_accuracies, best_val_acc


def save_angular_model(model, class_mapping, idx_to_class, accuracy, results):
    """Save the trained model for angular view detection"""
    print(f"\nSaving Angular View Model...")
    
    os.makedirs("models", exist_ok=True)
    
    # Save PyTorch model
    torch.save(model.state_dict(), "models/angular_cnn.pth")
    print("PyTorch model saved: models/angular_cnn.pth")
    
    # Export to ONNX format
    try:
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224)
        
        torch.onnx.export(
            model,
            dummy_input,
            "models/disease_cls.onnx",
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print("ONNX model saved: models/disease_cls.onnx")
    except Exception as e:
        print(f"ONNX export failed: {e}")
    
    # Save metadata
    metadata = {
        "class_mapping": class_mapping,
        "idx_to_class": idx_to_class,
        "num_classes": len(class_mapping),
        "accuracy": accuracy,
        "model_architecture": "MobileNetV2",
        "input_size": [224, 224],
        "optimization": "angular_view",
        "results": results
    }
    
    with open("models/cnn_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print("Metadata saved: models/cnn_metadata.json")
    
    return metadata


def main():
    """Main training function for angular view detection"""
    print("ANGULAR VIEW CNN TRAINING - MobileNetV2")
    print("=" * 70)
    print("Optimized for crawler camera angles")
    print("Estimated time: 2-3 hours")
    print("=" * 70)
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    start_time = time.time()
    
    try:
        # Load datasets
        image_paths, labels, class_mapping, idx_to_class = load_datasets_optimized()
        
        if len(image_paths) == 0:
            print("No images loaded. Check dataset paths.")
            return
        
        # Create data loaders
        train_loader, val_loader = create_angular_data_loaders(image_paths, labels, batch_size=16)
        
        # Create model
        num_classes = len(class_mapping)
        model = AngularPlantDiseaseCNN(num_classes)
        print(f"\n🏗️ Model created with {num_classes} classes")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train model
        trained_model, train_losses, val_losses, val_accuracies, best_acc = train_angular_model(
            model, train_loader, val_loader, num_epochs=15
        )
        
        # Save model
        results = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies,
            "best_accuracy": best_acc,
            "num_classes": num_classes,
            "total_images": len(image_paths),
            "optimization": "angular_view",
            "model_architecture": "MobileNetV2"
        }
        
        metadata = save_angular_model(
            trained_model, class_mapping, idx_to_class, best_acc, results
        )
        
        total_time = time.time() - start_time
        
        print(f"\nANGULAR VIEW TRAINING COMPLETED!")
        print("=" * 70)
        print(f"Total Time: {total_time/60:.1f} minutes")
        print(f"Best Accuracy: {best_acc:.2f}%")
        print(f"Total Classes: {num_classes}")
        print(f"Model: MobileNetV2 (Angular View Optimized)")
        print(f"ONNX Model: models/disease_cls.onnx")
        print(f"Metadata: models/cnn_metadata.json")
        print("Ready for crawler deployment!")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

