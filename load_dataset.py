import torch
from torch.utils.data import Dataset

class MockDataset(Dataset):
    def __init__(self, length=1250, image_shape=(1, 60, 60)):
        self.length = length
        self.image_shape = image_shape

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = torch.randn(self.image_shape)  # Random input
        y = torch.randn(self.image_shape)  # Random target
        return x, y

def load_dataset():
    return MockDataset()

if __name__=='__main__':
    print('End of Example')




if __name__=='__main__':
    print('End of Example')