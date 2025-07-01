# %%

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np

class RotationTransform:
    """Apply random rotation (0, 90, 180, 270 degrees) to image"""
    
    def __init__(self):
        self.rotations = [0, 90, 180, 270]
    
    def __call__(self, img, rotation=None):
        if rotation is None:
            rotation = np.random.choice(self.rotations)
        
        # Convert rotation angle to rotation index
        rotation_idx = self.rotations.index(rotation)
        
        # Apply rotation using flip and transpose operations (as in paper)
        if rotation == 0:
            rotated_img = img
        elif rotation == 90:
            # Transpose then flip vertically
            rotated_img = torch.flip(img.transpose(-2, -1), [-2])
        elif rotation == 180:
            # Flip vertically then horizontally
            rotated_img = torch.flip(torch.flip(img, [-2]), [-1])
        elif rotation == 270:
            # Flip vertically then transpose
            rotated_img = torch.flip(img, [-2]).transpose(-2, -1)
        
        return rotated_img, rotation_idx

class RotNetDataset(Dataset):
    """Dataset that applies rotations and returns rotation labels"""
    
    def __init__(self, base_dataset, return_all_rotations=True):
        self.base_dataset = base_dataset
        self.return_all_rotations = return_all_rotations
        self.rotation_transform = RotationTransform()
        
    def __len__(self):
        if self.return_all_rotations:
            return len(self.base_dataset) * 4  # 4 rotations per image
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        if self.return_all_rotations:
            # Return all 4 rotations of the same image (as mentioned in paper)
            base_idx = idx // 4
            rotation_idx = idx % 4
            rotation_angle = [0, 90, 180, 270][rotation_idx]
            
            img, _ = self.base_dataset[base_idx]
            rotated_img, rot_label = self.rotation_transform(img, rotation_angle)
            
            return rotated_img, rot_label
        else:
            # Return single random rotation
            img, _ = self.base_dataset[idx]
            rotated_img, rot_label = self.rotation_transform(img)
            return rotated_img, rot_label

class RotNet(nn.Module):
    """RotNet model for rotation prediction"""
    
    def __init__(self, backbone='resnet18', num_rotations=4):
        super(RotNet, self).__init__()
        
        # Use a standard backbone (simplified AlexNet-style for demo)
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv Block 2
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv Block 3
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            
            # Conv Block 4
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Conv Block 5
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # Classifier for rotation prediction
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_rotations),
        )
        
    def forward(self, x):
        features = self.features(x)
        x = features.view(features.size(0), -1)
        rotation_pred = self.classifier(x)
        return rotation_pred
    
    def get_features(self, x, layer='conv5'):
        """Extract features from intermediate layers"""
        if layer == 'conv5':
            return self.features(x)
        # Add more layer options as needed
        return self.features(x)

def train_rotnet(model, dataloader, num_epochs=100, device='cuda'):
    """Training function for RotNet"""
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    
    # Learning rate schedule (as in paper)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.2)
    
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, rotation_labels) in enumerate(dataloader):
            data, rotation_labels = data.to(device), rotation_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, rotation_labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += rotation_labels.size(0)
            correct += predicted.eq(rotation_labels).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, '
                      f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
        
        scheduler.step()
        print(f'Epoch {epoch} completed. Avg Loss: {running_loss/len(dataloader):.4f}, '
              f'Accuracy: {100.*correct/total:.2f}%')

def extract_features_for_downstream(model, dataloader, device='cuda'):
    """Extract features for downstream tasks"""
    
    model.eval()
    features_list = []
    labels_list = []
    
    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device)
            features = model.get_features(data)
            features = features.view(features.size(0), -1)
            
            features_list.append(features.cpu())
            labels_list.append(labels)
    
    return torch.cat(features_list, dim=0), torch.cat(labels_list, dim=0)

# Example usage:
if __name__ == "__main__":
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = RotNet()
    
    # Example with dummy data (replace with actual dataset)
    from torchvision.datasets import CIFAR10
    
    # Base transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Create base dataset
    base_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # Create rotation dataset
    rotation_dataset = RotNetDataset(base_dataset, return_all_rotations=True)
    dataloader = DataLoader(rotation_dataset, batch_size=128, shuffle=True, num_workers=4)
    
    # Train the model
    print("Training RotNet...")
    train_rotnet(model, dataloader, num_epochs=10, device=device)
    
    # After training, extract features for downstream tasks
    print("Extracting features for downstream tasks...")
    test_dataset = CIFAR10(root='./data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    features, labels = extract_features_for_downstream(model, test_loader, device)
    print(f"Extracted features shape: {features.shape}")
    
    # Now you can train a classifier on these features for the actual task
    # (object classification, detection, etc.)
