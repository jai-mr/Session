import torch
import torch.nn as nn
import torch.nn.functional as F

dropout = 0.05

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
        attn = torch.matmul(q, k.transpose(1, 0))
        attn = self.softmax(attn / (k.shape[-1]**0.5))
        z = torch.matmul(attn, v)
        z = self.fc_out(z)
        return z

    # def _am(self, q, k):
        # am = (q.transpose(1, 2) @ k) / self.sqrt_d_k

        # return F.softmax(am, dim=1)

    # def _z(self, v, am):
        # return v @ am

        
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
        
        