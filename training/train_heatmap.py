# train.py

import os
import json
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms

from heatmap_dataset import HeatmapDataset
# from custom_resnet import CustomResNet
from posenet import PoseNet

# -------------------------------
# Configuration and Hyperparameters
# -------------------------------

# Paths
TRAIN_IMAGES_DIR = 'heatmaps/test'

VAL_IMAGES_DIR = 'heatmaps/val'

SAVED_MODEL_PATH = 'saved_models/heatmap_cnn.pth'

# Hyperparameters
NUM_EPOCHS = 200
BATCH_SIZE = 5
LEARNING_RATE = 1e-3
NUM_WORKERS = 4  # Adjust based on your CPU
RANDOM_SEED = 42

BODY_PARTS = [
    "nose",
    "shoulder",
    "elbow",
    "wrist",
    "hip",
    "knee",
    "ankle",
    "nothing"
]

# -------------------------------
# Setting Random Seeds for Reproducibility
# -------------------------------

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True  # May slow down training

set_seed(RANDOM_SEED)


def save_model(model, path):
    """
    Saves the model state dictionary.
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# -------------------------------
# Training and Validation Loops
# -------------------------------

def train_model(model, device, train_loader, val_loader, criterion, optimizer, num_epochs, save_path):
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation", leave=False):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)

        epoch_val_loss = val_loss / len(val_loader.dataset)

        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {epoch_train_loss:.4f} - "
              f"Val Loss: {epoch_val_loss:.4f}")

        # Save the model if validation loss has decreased
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            save_model(model, save_path)

    print("Training complete.")

# -------------------------------
# Main Function
# -------------------------------

def main():
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((256, 192)),  # Resize to desired size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Using ImageNet means
                             std=[0.229, 0.224, 0.225])
    ])

    # Create Datasets
    train_dataset = HeatmapDataset(
        data_dir=TRAIN_IMAGES_DIR,
        heatmap_dir=TRAIN_IMAGES_DIR,
        transform=transform,
    )

    val_dataset = HeatmapDataset(
        data_dir=VAL_IMAGES_DIR,
        heatmap_dir=VAL_IMAGES_DIR,
        transform=transform,
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    # Initialize the model, loss function, and optimizer
    model = PoseNet(num_keypoints=7).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.7)

    # Ensure the save directory exists
    os.makedirs(os.path.dirname(SAVED_MODEL_PATH), exist_ok=True)

    # Train the model
    train_model(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=NUM_EPOCHS,
        save_path=SAVED_MODEL_PATH
    )

if __name__ == '__main__':
    main()
