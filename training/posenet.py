import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    """
    Implements a depthwise separable convolution.
    """
    def __init__(self, in_channels, out_channels, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride, 
            padding=padding, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, 
            padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out = self.depthwise(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.pointwise(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out

class MobileNetV1BackboneMinimalDownsampling(nn.Module):
    """
    MobileNetV1 Backbone with minimal downsampling (factor of 2).
    """
    def __init__(self):
        super(MobileNetV1BackboneMinimalDownsampling, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),  # Stride=2
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # All subsequent layers have stride=1 to maintain spatial dimensions
        self.layers = nn.Sequential(
            DepthwiseSeparableConv(32, 64, stride=1),
            DepthwiseSeparableConv(64, 128, stride=1),
            DepthwiseSeparableConv(128, 128, stride=1),
            DepthwiseSeparableConv(128, 256, stride=1),
            DepthwiseSeparableConv(256, 256, stride=1),
            DepthwiseSeparableConv(256, 512, stride=1),
            DepthwiseSeparableConv(512, 512, stride=1),
            DepthwiseSeparableConv(512, 512, stride=1),
            DepthwiseSeparableConv(512, 512, stride=1),
            DepthwiseSeparableConv(512, 512, stride=1),
            DepthwiseSeparableConv(512, 1024, stride=1),
            DepthwiseSeparableConv(1024, 1024, stride=1),
        )
        
    def forward(self, x):
        x = self.initial(x)  # Downsampled by factor of 2
        x = self.layers(x)   # No further downsampling
        return x  # Feature maps with spatial dimensions 128x96


class PoseNet(nn.Module):
    """
    PoseNet Model for heatmap-only training.
    """
    def __init__(self, num_keypoints=7):
        super(PoseNet, self).__init__()
        self.backbone = MobileNetV1BackboneMinimalDownsampling()
        
        # Define the heatmap head
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, num_keypoints, kernel_size=1)
        )
        
    def forward(self, x):
        # Forward pass through the backbone and heatmap head
        features = self.backbone(x)
        heatmaps = self.heatmap_head(features)
        return heatmaps



if __name__ == "__main__":
    # Define input dimensions
    batch_size = 1
    input_channels = 3
    input_height = 256
    input_width = 192
    
    # Create a PoseNet model instance
    model = PoseNet(num_keypoints=17)
    
    # Print the model architecture
    print(model)
    
    # Create a dummy input tensor
    input_tensor = torch.randn(batch_size, input_channels, input_height, input_width)
    
    # Forward pass
    heatmaps, offsets = model(input_tensor)
    
    # Print output shapes
    print(f"Input shape: {input_tensor.shape}")            # Expected: [1, 3, 256, 192]
    print(f"Heatmaps shape: {heatmaps.shape}")            # Expected: [1, 17, 128, 96]
    print(f"Offsets shape: {offsets.shape}")              # Expected: [1, 34, 128, 96]
