import torch
import torch.nn as nn
import torch.nn.functional as F

class ThreeLayerCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        return x
    

class DiffNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, meta_channels=5):
        super(DiffNet, self).__init__()
        self.layers = [ThreeLayerCNN(in_channels, meta_channels) for _ in range(3)]
        self.dt = nn.Parameter(torch.tensor(0.1), requires_grad=True)  # learnable weight dt
    
    def diff_update(self, layer, x):
        meta = layer(x)
        
        rolled_up    = self.shift_and_pad(x,  1, 2)
        rolled_down  = self.shift_and_pad(x, -1, 2)
        rolled_left  = self.shift_and_pad(x, -1, 3)
        rolled_right = self.shift_and_pad(x,  1, 3)
        
        update_sequence = zip(
            [rolled_up, rolled_down, rolled_left, rolled_right, x],
            [meta[:, i, :, :].unsqueeze(1) for i in range(meta.shape[1])]
        )
        update_list = [m * r for r, m in update_sequence]
        update = torch.stack(update_list, dim=0).sum(dim=0)
        return x + self.dt * update

    def shift_and_pad(self, x, shift, dim):
        if shift == 0:
            return x
        pad = [0, 0, 0, 0]
        pad[2 * (3 - dim) + (0 if shift > 0 else 1)] = abs(shift)
        x_padded = F.pad(x, pad, mode='constant', value=0)
        if shift > 0:
            return x_padded.narrow(dim, 0, x.size(dim))
        else:
            return x_padded.narrow(dim, abs(shift), x.size(dim))

    def forward(self, x):
        for layer in self.layers:
            x = self.diff_update(layer, x)
            
        return F.relu(x)





