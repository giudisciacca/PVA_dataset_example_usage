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
    def __init__(self, path, image_shape=(1, 60, 60), upfront_load = True, input_data_string='imagesDiff', target_data_string='imagesInput'):
        self.path = path
        
        self.image_shape = image_shape
        self.upfront_load = upfront_load
        if self.upfront_load: 
            self._input_data_full = self.load_h5py(self.path, input_data_string)
            self._target_data_full = self.load_h5py(self.path, target_data_string)
        else:
            self._input_data_full = None
            self._target_data_full = None
        
        if self._input_data_full is not None and self._target_data_full is not None:
            self.length = self._input_data_full.shape[0]    
    
    def __add__(self, other):
        if other.upfront_load != True or self.upfront_load != True:
            raise ValueError("Both datasets must have upfront_load set to True to be added.")
        if self.path == other.path:
            raise ValueError("Cannot add datasets with the same path.")
        # Concatenate input and target arrays
        input_data_full = np.concatenate([self._input_data_full, other._input_data_full], axis=0)
        target_data_full = np.concatenate([self._target_data_full, other._target_data_full], axis=0)

        # Create a new dataset instance
        new_dataset = PVADataset(self.path + "+" + other.path, image_shape=self.image_shape, upfront_load=False)
        new_dataset._input_data_full = input_data_full
        new_dataset._target_data_full = target_data_full
        new_dataset.length = input_data_full.shape[0]
        new_dataset.upfront_load = True
        return new_dataset    
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx, preprocess_function=identity): 
        if self._input_data_full is None or self._target_data_full is None:
            x = self.load_h5py(self.path, 'imagesDiff')[idx]
            y = self.load_h5py(self.path, 'imagesInput')[idx]
        else:
            x = self._input_data_full[idx] 
            y = self._target_data_full[idx] 
        x = preprocess_function(x)
        y = preprocess_function(y)
        return torch.tensor(x[...,0]).unsqueeze(0),torch.tensor(y[...,0]).unsqueeze(0)
    
    def load_h5py(self, path, field, idx = None):
        with h5py.File(path, 'r') as f:
            if idx is not None:
                data = f[field][idx]
            else:
                data = f[field][:]
        return data  
if __name__=='__main__':
    print('End of Example')
