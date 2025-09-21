"""
Unified CNN Training Script - MobileNetV2 for Multi-Crop Disease Detection
Trains a single CNN model on all 20 classes (potato + tomato + pepper bell)
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


class PlantDiseaseDataset(Dataset):
    """Custom dataset for plant disease images"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def load_all_datasets():
    """Load images from all three crop datasets"""
    print("Loading All Crop Datasets...")
    print("=" * 60)
    
    all_images = []
    all_labels = []
    class_mapping = {}
    
    # Dataset paths
    datasets = {
        'potato': r"C:\Users\yashv.HPLAPTOP\OneDrive\Documents\data)vijay\potato_split\kaggle\working\data\potato",
        'tomato': r"C:\Users\yashv.HPLAPTOP\OneDrive\Documents\data)vijay\tomato_split\tomato_split\train",
        'pepper': r"C:\Users\yashv.HPLAPTOP\OneDrive\Documents\data)vijay\pepper_split\pepper_split\train"
    }
    
    class_idx = 0
    
    for crop_name, dataset_path in datasets.items():
        print(f"\n📁 Loading {crop_name.upper()} dataset...")
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            print(f"Dataset not found: {dataset_path}")
            continue
        
        # Get class directories
        if crop_name == 'potato':
            class_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
        else:
            class_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
        
        crop_classes = [d.name for d in class_dirs]
        print(f"Found {len(crop_classes)} classes: {crop_classes}")
        
        for class_name in crop_classes:
            class_dir = dataset_path / class_name
            print(f"  Loading {class_name}...")
            
            # Get all image files
            image_files = list(class_dir.glob('*.JPG')) + list(class_dir.glob('*.jpg'))
            
            # Limit images per class for balanced training
            max_images = 500 if crop_name == 'tomato' else 300
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


def create_data_loaders(image_paths, labels, batch_size=32, train_split=0.8):
    """Create train and validation data loaders"""
    print(f"\nCreating data loaders...")
    
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
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = PlantDiseaseDataset(train_images, train_labels, train_transform)
    val_dataset = PlantDiseaseDataset(val_images, val_labels, val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader


class PlantDiseaseCNN(nn.Module):
    """MobileNetV2-based plant disease classifier"""
    
    def __init__(self, num_classes):
        super(PlantDiseaseCNN, self).__init__()
        
        # Load pretrained MobileNetV2
        self.backbone = mobilenet_v2(pretrained=True)
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.backbone.last_channel, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


def train_model(model, train_loader, val_loader, num_epochs=20, learning_rate=0.001):
    """Train the CNN model"""
    print(f"\nTraining CNN Model...")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Training history
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (images, labels) in enumerate(train_pbar):
            images, labels = images.to(device), labels.to(device)
            
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
                images, labels = images.to(device), labels.to(device)
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
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        # Store history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Best Val Acc: {best_val_acc:.2f}%")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses, val_accuracies, best_val_acc


def evaluate_model(model, val_loader, idx_to_class):
    """Evaluate the trained model"""
    print(f"\nEvaluating Model...")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"Test Accuracy: {accuracy:.3f}")
    
    # Classification report
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    report = classification_report(all_labels, all_predictions, target_names=class_names)
    print(f"\nClassification Report:")
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('CNN Model Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('cnn_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Confusion matrix saved to cnn_confusion_matrix.png")
    
    return accuracy, all_predictions, all_labels, all_probabilities


def save_model_and_metadata(model, class_mapping, idx_to_class, accuracy, results):
    """Save the trained model and metadata"""
    print(f"\nSaving Model and Metadata...")
    
    os.makedirs("models", exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), "models/unified_cnn.pth")
    print("Model saved: models/unified_cnn.pth")
    
    # Save class mappings
    metadata = {
        "class_mapping": class_mapping,
        "idx_to_class": idx_to_class,
        "num_classes": len(class_mapping),
        "accuracy": accuracy,
        "model_architecture": "MobileNetV2",
        "input_size": [224, 224],
        "results": results
    }
    
    with open("models/cnn_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print("Metadata saved: models/cnn_metadata.json")
    
    return metadata


def main():
    """Main training function"""
    print("UNIFIED CNN TRAINING - MobileNetV2")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    try:
        # Load all datasets
        image_paths, labels, class_mapping, idx_to_class = load_all_datasets()
        
        if len(image_paths) == 0:
            print("No images loaded. Check dataset paths.")
            return
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(image_paths, labels, batch_size=32)
        
        # Create model
        num_classes = len(class_mapping)
        model = PlantDiseaseCNN(num_classes)
        print(f"\n🏗️ Model created with {num_classes} classes")
        
        # Train model
        trained_model, train_losses, val_losses, val_accuracies, best_acc = train_model(
            model, train_loader, val_loader, num_epochs=20
        )
        
        # Evaluate model
        accuracy, predictions, true_labels, probabilities = evaluate_model(
            trained_model, val_loader, idx_to_class
        )
        
        # Save model and metadata
        results = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies,
            "best_accuracy": best_acc,
            "final_accuracy": accuracy,
            "num_classes": num_classes,
            "total_images": len(image_paths)
        }
        
        metadata = save_model_and_metadata(
            trained_model, class_mapping, idx_to_class, accuracy, results
        )
        
        print(f"\nTRAINING COMPLETED!")
        print("=" * 60)
        print(f"Final Accuracy: {accuracy:.3f}")
        print(f"Best Validation Accuracy: {best_acc:.2f}%")
        print(f"Total Classes: {num_classes}")
        print(f"Model Architecture: MobileNetV2")
        print(f"Model saved: models/unified_cnn.pth")
        print("Ready for deployment!")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

