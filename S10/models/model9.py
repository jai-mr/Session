import torch
import torch.nn as nn
import torch.nn.functional as F

'''
The UltimusBlock class defines a single self-attention block. 
The block takes an input tensor and applies three linear transformations (fc_k, fc_q, and fc_v) to project the input into three different spaces. 
The resulting key, query, and value tensors are then used to compute an attention matrix(AM) using matrix multiplication and a softmax activation function. 
The attention matrix is then used to weight the value tensor and compute a weighted sum 
This is then projected back into the original space using another linear transformation (fc_out).
'''
class UltimusBlock(nn.Module):
    def __init__(self, in_channels):
        super(UltimusBlock, self).__init__()
        self.fc_k = nn.Linear(in_channels, in_channels//6)
        self.fc_q = nn.Linear(in_channels, in_channels//6)
        self.fc_v = nn.Linear(in_channels, in_channels//6)
        self.softmax = nn.Softmax(dim=-1)
        self.fc_out = nn.Linear(in_channels//6, in_channels)
        
    def forward(self, x):
        k = self.fc_k(x)
        q = self.fc_q(x)
        v = self.fc_v(x)
        AM = torch.matmul(q, k.transpose(1, 0))
        AM = self.softmax(AM / (k.shape[-1]**0.5))
        z = torch.matmul(AM, v)
        z = self.fc_out(z)
        return z
'''
The UltimusNet class defines the entire neural network.
The first three convolutional layers use a kernel size of 3x3 with padding of 1
They progressively increase the number of output channels from 16 to 32 to 48, which increases the depth and complexity of the learned features
The adaptive average pooling layer reduces the spatial dimensions of the output of the last convolutional layer to a size of 1x1. 
This effectively collapses the spatial information and retains only the channel information, which can be used to capture global context of the input.
The adaptive average pooling layer reduces the spatial dimensions of the output of the last convolutional layer to a size of 1x1. 
This effectively collapses the spatial information and retains only the channel information, which can be used to capture global context of the input.
The output of the last Ultimus block is passed through a linear layer with 10 output units, which corresponds to the number of classes in the classification task
'''
     
class UltimusNet(nn.Module):
    def __init__(self):
        super(UltimusNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 48, kernel_size=3, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.ultimus1 = UltimusBlock(48)
        self.ultimus2 = UltimusBlock(48)
        self.ultimus3 = UltimusBlock(48)
        self.ultimus4 = UltimusBlock(48)
        self.fc = nn.Linear(48, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.ultimus1(x)
        x = self.ultimus2(x)
        x = self.ultimus3(x)
        x = self.ultimus4(x)
        x = self.fc(x)
        return x
        
class TransformerModel(nn.Module):
    def __init__(self, num_classes=10, n_features=48):
        super(TransformerModel, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.ultimus_blocks = nn.Sequential(
            UltimusBlock(in_channels=48),
            UltimusBlock(in_channels=48),
            UltimusBlock(in_channels=48),
            UltimusBlock(in_channels=48),
        )

        self.final_fc = nn.Linear(
            in_features=n_features, out_features=num_classes, bias=False
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.gap(x)

        # Reshape [batch_size, 48, 1, 1] to [batch_size, 48]
        x = x.view(-1, 48)

        x = self.ultimus_blocks(x)

        out = self.final_fc(x)

        # Reshape to [batch_size, 10]
        out = out.view(out.size(0), -1)
        return out
