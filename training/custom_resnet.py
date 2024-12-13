import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the Bottleneck Residual Block
class BottleneckBlock(nn.Module):
    expansion = 4  # Expansion factor for output channels

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        Initializes the BottleneckBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels before expansion.
            stride (int): Stride for the convolution.
            downsample (nn.Module or None): Downsampling layer to match dimensions.
        """
        super(BottleneckBlock, self).__init__()
        
        # 1x1 convolution to reduce channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                               kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 3x3 convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 1x1 convolution to expand channels
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        
        # First convolution
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Second convolution
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        # Third convolution
        out = self.conv3(out)
        out = self.bn3(out)
        
        # Downsample if needed
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Add residual connection
        out += identity
        out = self.relu(out)
        
        return out

# Enhanced Custom ResNet with Bottleneck Blocks
class CustomResNet(nn.Module):
    def __init__(self, num_heatmap_channels=7, block=BottleneckBlock, layers=[3, 4, 23, 3]):
        """
        Initializes the CustomResNet.

        Args:
            num_heatmap_channels (int): Number of output heatmap channels.
            block (nn.Module): Residual block type (BottleneckBlock).
            layers (list): Number of blocks in each of the 4 layers.
        """
        super(CustomResNet, self).__init__()
        
        self.in_channels = 64
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, self.in_channels, 
                               kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Max pooling layer (standard in ResNet)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)   # 256 channels after expansion
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)   # 512 channels after expansion
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)   # 1024 channels after expansion
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)   # 2048 channels after expansion
        
        # Final convolution to get heatmaps
        self.final_conv = nn.Conv2d(512 * block.expansion, num_heatmap_channels, 
                                    kernel_size=1, stride=1, padding=0)
        
        # Upsampling layers to reach desired resolution (128x96)
        # Calculate the downsampling factor:
        # Initial conv (stride=2) -> 128x96
        # MaxPool (stride=2) -> 64x48
        # Layer1 (stride=1) -> 64x48
        # Layer2 (stride=2) -> 32x24
        # Layer3 (stride=2) -> 16x12
        # Layer4 (stride=2) -> 8x6
        # Final conv -> 8x6
        # To reach 128x96, need to upsample by 16x
        # We'll use multiple upsampling steps with convolutional refinement

        self.upsample_full = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 8 -> 16
            nn.Conv2d(num_heatmap_channels, num_heatmap_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_heatmap_channels),
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 16 -> 32
            nn.Conv2d(num_heatmap_channels, num_heatmap_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_heatmap_channels),
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 32 -> 64
            nn.Conv2d(num_heatmap_channels, num_heatmap_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_heatmap_channels),
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 64 -> 128
            nn.Conv2d(num_heatmap_channels, num_heatmap_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_heatmap_channels),
            nn.ReLU(inplace=True),
            
            # Optional: Additional upsample if needed for exact size
            nn.Upsample(size=(128, 96), mode='bilinear', align_corners=True)
        )
        
        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, block, out_channels, blocks, stride=1):
        """
        Creates a residual layer with a specified number of blocks.

        Args:
            block (nn.Module): Residual block type.
            out_channels (int): Number of output channels before expansion.
            blocks (int): Number of residual blocks.
            stride (int): Stride for the first block.

        Returns:
            nn.Sequential: Sequential container of residual blocks.
        """
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            # Downsample to match dimensions
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, 
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """
        Initializes weights using Kaiming He initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 256, 192).

        Returns:
            torch.Tensor: Heatmap tensor of shape (batch_size, 7, 128, 96).
        """
        # Input: (batch_size, 3, 256, 192)
        x = self.conv1(x)       # (batch_size, 64, 128, 96)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)     # (batch_size, 64, 64, 48)
        
        x = self.layer1(x)      # (batch_size, 256, 64, 48)
        x = self.layer2(x)      # (batch_size, 512, 32, 24)
        x = self.layer3(x)      # (batch_size, 1024, 16, 12)
        x = self.layer4(x)      # (batch_size, 2048, 8, 6)
        
        x = self.final_conv(x)  # (batch_size, 7, 8, 6)
        x = self.upsample_full(x)  # (batch_size, 7, 128, 96)
        
        return x

# Example usage:
if __name__ == "__main__":
    model = CustomResNet(num_heatmap_channels=7, layers=[3, 4, 23, 3])  # Similar to ResNet-101
    input_tensor = torch.randn(1, 3, 256, 192)
    output = model(input_tensor)
    print(output.shape)  # Expected: torch.Size([1, 7, 128, 96])
