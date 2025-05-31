import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

def load_dataset(**kwargs):
    """
    Returns an instance of PVADataset with the provided keyword arguments.
    """
    return PVADataset(**kwargs)

def rgb2gray(rgb):
    """
    Convert an RGB image to grayscale using standard luminance conversion.
    Args:
        rgb (np.ndarray): Input RGB image.
    Returns:
        np.ndarray: Grayscale image.
    """
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def identity(data):
    """
    Identity preprocessing function. Converts data to float32.
    Args:
        data (np.ndarray): Input data.
    Returns:
        np.ndarray: Data as float32.
    """
    return data.astype('float32')

def preprocess_norm_by_max(data):       
    """
    Convert RGB image to grayscale and normalize by its maximum value.
    Args:
        data (np.ndarray): Input RGB image.
    Returns:
        np.ndarray: Normalized grayscale image as float32.
    """
    data = rgb2gray(data) 
    data = data/data.max()
    return data.astype('float32')

class PVADataset(Dataset):
    """
    PyTorch Dataset for loading PVA HDF5 data with optional upfront loading and preprocessing.
    """
    def __init__(self, path, image_shape=(1, 60, 60), upfront_load = True, input_data_string='imagesDiff', target_data_string='imagesInput'):
        """
        Args:
            path (str): Path to the HDF5 file.
            image_shape (tuple): Shape of each image (C, H, W).
            upfront_load (bool): If True, load all data into memory at initialization.
            input_data_string (str): Field name for input data in HDF5 file.
            target_data_string (str): Field name for target data in HDF5 file.
        """
        self.path = path
        
        allowed_fields = {'imagesDiff', 'imagesInput', 'imagesControl', 'imagesInputControl'}
        if input_data_string not in allowed_fields:
            raise ValueError(f"input_data_string must be one of {allowed_fields}, got '{input_data_string}'")
        if target_data_string not in allowed_fields:
            raise ValueError(f"target_data_string must be one of {allowed_fields}, got '{target_data_string}'")

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
        """
        Concatenate two PVADataset instances (with upfront_load=True) along the first dimension.
        Args:
            other (PVADataset): Another PVADataset instance.
        Returns:
            PVADataset: New dataset with concatenated data.
        """
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
        """
        Returns the number of samples in the dataset.
        """
        return self.length

    def __getitem__(self, idx, preprocess_function=identity): 
        """
        Get a sample from the dataset, optionally applying a preprocessing function.
        Args:
            idx (int): Index of the sample.
            preprocess_function (callable): Function to preprocess the data.
        Returns:
            tuple: (input_tensor, target_tensor)
        """
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
        """
        Load data from an HDF5 file.
        Args:
            path (str): Path to the HDF5 file.
            field (str): Field name in the HDF5 file.
            idx (int, optional): Index to load a single sample. If None, load all data.
        Returns:
            np.ndarray: Loaded data.
        """
        with h5py.File(path, 'r') as f:
            if idx is not None:
                data = f[field][idx]
            else:
                data = f[field][:]
        return data  

if __name__=='__main__':
    print('End of Example')