import torch
import torch.nn as nn
import torch.nn.functional as F

class FourLayerCNN(nn.Module):
    def __init__(self, in_channels, out_channels, cnn_channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, cnn_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(cnn_channels)
        self.pool1 = nn.MaxPool2d(2, stride=1, padding=1)
        self.conv2 = nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(cnn_channels)
        self.pool2 = nn.MaxPool2d(2, stride=1, padding=1)
        self.conv3 = nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(cnn_channels)
        self.pool3 = nn.MaxPool2d(2, stride=1, padding=1)
        self.conv4 = nn.Conv2d(cnn_channels, out_channels, kernel_size=3, padding=1)

    def center_crop(self, x, target_shape):
        _, _, h, w = x.shape
        th, tw = target_shape
        i = (h - th) // 2
        j = (w - tw) // 2
        return x[:, :, i:i+th, j:j+tw]

    def forward(self, x):
        input_shape = x.shape[2:]
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.conv4(x)
        # Center crop to match input spatial dimensions
        if x.shape[2:] != input_shape:
            x = self.center_crop(x, input_shape)
        return F.relu(x)
    
class DiffNet(nn.Module):
    def __init__(self, in_channels=1, meta_channels=5, num_layers=5):
        super().__init__()
        self.layers = nn.ModuleList([FourLayerCNN(in_channels, meta_channels) for _ in range(num_layers)])
        self.dt = nn.Parameter(torch.tensor(0.1), requires_grad=True)  # learnable weight dt
    
    def diff_update(self, layer, x):
        meta = layer(x)
        
        rolled_up    = self.shift_and_pad(x,  1, 2)
        rolled_down  = self.shift_and_pad(x, -1, 2)
        rolled_left  = self.shift_and_pad(x, -1, 3)
        rolled_right = self.shift_and_pad(x,  1, 3)
        update_sequence = zip(
            [-x, rolled_up, rolled_down, rolled_left, rolled_right],
            [meta[:, i, :, :].unsqueeze(1) for i in range(meta.shape[1])]
        )
        update_list = [ torch.multiply(m , r) for r, m in update_sequence]
        update = torch.stack(update_list, dim=0).sum(dim=0)
        return x - self.dt*update

    def shift_and_pad(self, x, shift, dim):
        if shift == 0:
            return x
        pad = [abs(shift), abs(shift), abs(shift), abs(shift)]
        x_padded = F.pad(x, pad, mode='constant', value=0)
        if shift !=0:
            return torch.roll(x_padded, shift, dim)[...,abs(shift):-abs(shift),abs(shift):-abs(shift)]
        
    def forward(self, x):
        for layer in self.layers:
            x = self.diff_update(layer, x)
        x = F.relu(x)  
        return x





