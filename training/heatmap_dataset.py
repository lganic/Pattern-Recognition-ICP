import os
import numpy as np
from glob import glob
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class HeatmapDataset(Dataset):
    def __init__(self, data_dir, heatmap_dir, transform=None, heatmap_transform=None):
        """
        Args:
            data_dir (str): Directory with images.
            heatmap_dir (str): Directory with heatmap .npy files.
            transform (callable, optional): Optional transform to be applied on an image.
            heatmap_transform (callable, optional): Optional transform to be applied on a heatmap.
        """
        self.data_dir = data_dir
        self.heatmap_dir = heatmap_dir
        self.transform = transform
        self.heatmap_transform = heatmap_transform

        # Assuming that each image has a corresponding heatmap with the same UUID
        # For example: image_001.png <-> image_001_heatmap.npy
        self.image_files = sorted(glob(os.path.join(data_dir, '*.jpg')))  # Adjust the pattern if needed
        self.heatmap_files = sorted(glob(os.path.join(heatmap_dir, '*_heatmap.npy')))

        # Extract UUIDs to ensure matching
        self.image_uuids = [os.path.splitext(os.path.basename(f))[0] for f in self.image_files]
        self.heatmap_uuids = [os.path.splitext(os.path.basename(f))[0].replace('_heatmap', '') for f in self.heatmap_files]

        # Create a mapping from UUID to heatmap file
        self.uuid_to_heatmap = {uuid: heatmap for uuid, heatmap in zip(self.heatmap_uuids, self.heatmap_files)}

        # Filter image files that have corresponding heatmaps
        self.filtered_image_files = [f for f in self.image_files if os.path.splitext(os.path.basename(f))[0] in self.uuid_to_heatmap]

    def __len__(self):
        return len(self.filtered_image_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.filtered_image_files[idx]
        uuid = os.path.splitext(os.path.basename(img_path))[0]
        heatmap_path = self.uuid_to_heatmap[uuid]

        # Load image
        image = Image.open(img_path).convert('RGB')  # Ensure image is in RGB

        # Apply image transformations
        if self.transform:
            image = self.transform(image)

        # Load heatmap
        heatmap = np.load(heatmap_path)  # Shape: (H, W, C)

        # assert heatmap.shape == (96, 128, 7), f"Unexpected heatmap shape: {heatmap.shape}"

        # # Debug: Print original heatmap shape
        # print(f"Original heatmap shape (W, H, C): {heatmap.shape}")

        # Check and transpose if necessary
        if heatmap.shape[0] != 128 or heatmap.shape[1] != 96:
            heatmap = np.transpose(heatmap, (1, 0, 2))  # Swap W and H to (H, W, C)
            # print(f"Fixed heatmap shape (W, H, C): {heatmap.shape}")

        # Convert heatmap to tensor and permute to (C, H, W)
        heatmap = torch.from_numpy(heatmap).float().permute(2, 0, 1)
        
        # # Debug: Print permuted heatmap shape
        # print(f"Heatmap tensor shape (C, H, W): {heatmap.shape}")

        # Apply heatmap transformations if any
        if self.heatmap_transform:
            heatmap = self.heatmap_transform(heatmap)

        return image, heatmap

# Example usage
def get_dataloader(data_dir='heatmaps', batch_size=32, shuffle=True, num_workers=4):
    """
    Args:
        data_dir (str): Directory containing images and heatmaps.
        batch_size (int): How many samples per batch to load.
        shuffle (bool): Set to True to have the data reshuffled at every epoch.
        num_workers (int): How many subprocesses to use for data loading.
    Returns:
        DataLoader: PyTorch DataLoader for the dataset.
    """
    # Define paths
    images_dir = data_dir  # Assuming images are copied to 'heatmaps' as per the original script
    heatmaps_dir = data_dir  # Heatmaps are saved in 'heatmaps' directory

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to desired size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Using ImageNet means
                             std=[0.229, 0.224, 0.225])
    ])

    # Define heatmap transformations (if any)
    heatmap_transform = transforms.Compose([
        # Add any heatmap-specific transformations here
        # For example, normalization can be added if needed
    ])

    dataset = HeatmapDataset(
        data_dir=images_dir,
        heatmap_dir=heatmaps_dir,
        transform=transform,
        heatmap_transform=heatmap_transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True  # Useful if using GPU
    )

    return dataloader

# Example of iterating through the dataloader
if __name__ == "__main__":
    dataloader = get_dataloader()

    for batch_idx, (images, heatmaps) in enumerate(dataloader):
        print(f"Batch {batch_idx+1}")
        print(f"Images shape: {images.shape}")      # Expected: (batch_size, 3, 256, 256)
        print(f"Heatmaps shape: {heatmaps.shape}")  # Expected: (batch_size, C, 256, 256)
        # Add your training loop or evaluation code here
        break  # Remove this break to iterate through the entire dataset
