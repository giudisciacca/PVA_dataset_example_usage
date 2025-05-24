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
        rolled_up = torch.roll(x, shifts=(1,0), dims=(2, 3))
        rolled_down = torch.roll(x, shifts=(-1,0), dims=(2, 3))
        rolled_left = torch.roll(x, shifts=(0,-1), dims=(2, 3))
        rolled_right = torch.roll(x, shifts=(0,1), dims=(2, 3))
        update_sequence = zip([rolled_up, rolled_down, rolled_left, rolled_right,x],
                              [ meta[:,i,:,:]  for i in range(meta.shape[1])])
        return x +  self.dt * torch.sum(m*r for r,m in update_sequence ) 
    
    def forward(self, x):
        for layer in self.layers:
            x = self.diff_update(layer, x)
            
        return F.relu(x)





