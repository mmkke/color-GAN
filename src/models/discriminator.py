"""
Nelson Farrell & Michael Massone
Image Enhancement: Colorization - cGAN
CS 7180 Advanced Perception
Bruce Maxwell, PhD.
09-28-2024

This file cotains various implementations of discriminator networks
"""
########################################################## Packages ######################################################## 
import torch
from torch import nn, optim
import torch.nn.functional as F

################################################## Discriminator 1 #########################################################
class Discriminator(nn.Module):
    """
    This discrimintor is based on the work of Nazeri et al., but will modifications.
        1. Additional convolutional layers
        2. Fully connected linear later
    """
    def __init__(self):
        """
        Network initializer
        """
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 4, stride = 2)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 4, stride = 2)
        self.conv3 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 4, stride = 2)
        self.conv4 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 4, stride = 2)
        self.conv5 = nn.Conv2d(in_channels = 512, out_channels = 1024, kernel_size = 4, stride = 2)
        self.conv6 = nn.Conv2d(in_channels = 1024, out_channels = 1, kernel_size = 4, stride = 1)
        
        # Linear layer
        self.fc1 = nn.Linear(9, 1)

        # Batch norms
        self.batchnorm64 = nn.BatchNorm2d(64)
        self.batchnorm128 = nn.BatchNorm2d(128)
        self.batchnorm256 = nn.BatchNorm2d(256)
        self.batchnorm512 = nn.BatchNorm2d(512)
        self.batchnorm1024 = nn.BatchNorm2d(1024)

        # Leaky ReLU activation function
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        """
        Forward pass
        """
        # This will apply batch norm and leaky relu to the first 5 conv layers
        x = self.leaky_relu(self.batchnorm64(self.conv1(x)))
        x = self.leaky_relu(self.batchnorm128(self.conv2(x)))
        x = self.leaky_relu(self.batchnorm256(self.conv3(x)))
        x = self.leaky_relu(self.batchnorm512(self.conv4(x)))
        x = self.leaky_relu(self.batchnorm1024(self.conv5(x)))

        # The 6th conv, no leaky relu or batch norm
        x = self.conv6(x)

        # Fatten the tensor and apply linear later
        x = torch.flatten(x, 1)
        x = self.fc1(x)

        # Final activation to return probability
        x = torch.sigmoid(x)

        return x
        
################################################## Discriminator 2 #########################################################
class Discriminator_2(nn.Module):
    """
    This discrimintor is based on the work of Nazeri et al. without modifications. It is a fully convolutional
    neural network
    """
    def __init__(self):
        """
        Network initializer
        """
        super().__init__()

        # Conv layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=0)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=0)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=0)

        # Batch normalization layers
        self.batchnorm64 = nn.BatchNorm2d(64)
        self.batchnorm128 = nn.BatchNorm2d(128)
        self.batchnorm256 = nn.BatchNorm2d(256)
        self.batchnorm512 = nn.BatchNorm2d(512)


        # Leaky ReLU activation function
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        """
        Forward pass
        """
        # Apply conv, batch norm, and Leaky ReLU for each layer
        x = self.leaky_relu(self.batchnorm64(self.conv1(x)))
        x = self.leaky_relu(self.batchnorm128(self.conv2(x)))
        x = self.leaky_relu(self.batchnorm256(self.conv3(x)))
        x = self.leaky_relu(self.batchnorm512(self.conv4(x)))
        
        # Final convolution layer reduces to 1x1
        x = self.conv5(x)
        
        return x

################################################## PatchDiscriminator #########################################################
class PatchDiscriminator(nn.Module):
    """
    This is a patch discriminator based on the work of Isola et al.
    """
    def __init__(self, input_channels=3):
        """
        Model initializer
        """
        super(PatchDiscriminator, self).__init__()
        
        # Define the sequential layers
        self.model = nn.Sequential(
            # Block 1: Input to 64 channels, no batch norm in the first block
            nn.Sequential(
                nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            # Block 2: 64 -> 128 channels
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            # Block 3: 128 -> 256 channels
            nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            # Block 4: 256 -> 512 channels
            nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True)
            ),
            # Output layer: 512 -> 1 channel (patch-level decision)
            nn.Sequential(
                nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)  # No batch norm or activation here
            )
        )
    
    def forward(self, x):
        """
        Forward pass
        """
        return self.model(x)

###################################################### End ###############################################################
if __name__ == "__main__":
    pass
