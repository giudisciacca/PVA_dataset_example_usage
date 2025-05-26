import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

def load_dataset(path):
    return PVADataset(path=path)


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
def identity(data):
    return data.astype('float32')

def preprocess_norm_by_max(data):       
    data = rgb2gray(data) 
    data = data/data.max()
    return data.astype('float32')

class PVADataset(Dataset):
    def __init__(self, path, image_shape=(1, 60, 60)):
        self.path = path
        with h5py.File(self.path, 'r') as f:
            self.length = len(f['imagesInput'])
        self.image_shape = image_shape

    def __len__(self):
        return self.length

    def __getitem__(self, idx, input_data_string ='imagesDiff', target_data_string='imagesInput'): 
        #path_x = f"{self.path}/{input_data_string}_{idx:05d}.h5"
        #path_y = f"{self.path}/{target_data_string}_{idx:05d}.h5"
        
        x = self.load_h5py(self.path, input_data_string, idx) 
        y = self.load_h5py(self.path, target_data_string, idx) 

        return x, y
    
    def load_h5py(self, path, field, idx, preprocess_function=identity):
        with h5py.File(path, 'r') as f:
            data = f[field][:]
        data = preprocess_function(data)
        return torch.tensor(data[idx,...,0]).unsqueeze(0)  
if __name__=='__main__':
    print('End of Example')
