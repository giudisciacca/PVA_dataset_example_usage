import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.utils as vutils

import torchvision.transforms as T
from PIL import Image
import os


from diffnet import DiffNet
from unet import UNet  # Assumes UNet is defined in unet.py
from load_dataset import load_dataset  # Assumes this returns a torch Dataset

# Hyperparameters
batch_size = 1
epochs = 10
learning_rate = 1e-4
patience = 10  # Early stopping patience
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
folder = '/workspace/pva-dataset/Folder/HDF5/'
file = '20210630_expsizes_EMNIST5_moving_phantom1_5mm_2FT_phantom2_5mm_1FT/20210630_expsizes_EMNIST5_moving_phantom1_5mm_2FT_phantom2_5mm_1FT/EXP2021_siz2_phantom1_5mm_2FT_phantom2_5mm_1FT_moving_normPeak'
file = '20210614_exp_EMNIST2_moving_phantom1_5mm_1FT_phantom2_5mm_1FT/EXP2021_phantom1_5mm_1FT_phantom2_5mm_1FT_moving_normPeak'
dataset = load_dataset(folder + file + '.h5')
dataset += load_dataset(folder + file + '_v_t.h5')
dataset += load_dataset(folder + file + '_t.h5')
n_total = len(dataset)
n_val = n_total // 10
n_test = n_total // 10
n_train = n_total - n_val - n_test

train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size)
test_loader = DataLoader(test_set, batch_size=batch_size)

# Model, loss, optimizer

for model in [ DiffNet(in_channels=1),UNet(n_channels=1, n_classes=1) ]:
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    def evaluate(loader):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)/torch.norm(y)
                total_loss += loss.item() 
        return total_loss / len(loader.dataset)

    # Training loop with early stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            #print(f"Epoch {epoch+1}/{epochs} - Training Loss: {loss.item():.4f}")
        val_loss = evaluate(val_loader)
        print(f"Epoch {epoch+1}/{epochs} - Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model._get_name()+"_best_model.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(model._get_name()+"_best_model.pt"))
    test_loss = evaluate(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    
    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(model._get_name()+"_best_model.pt"))
    test_loss = evaluate(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    
    model.eval()
    to_pil = T.ToPILImage()
    n_grid = 16
    input_images = []
    output_images = []
    target_images = []
    indices_used = []

    # Collect the first n_grid images and their outputs
    for x, y in test_loader:
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
        for i in range(x.size(0)):
            if len(input_images) >= n_grid:
                break
            input_img = x[i].cpu()
            target_img = y[i].cpu()
            output_img = pred[i].cpu()
            # If single channel, squeeze to (H, W)
            if input_img.shape[0] == 1:
                input_img = input_img.squeeze(0)
            if output_img.shape[0] == 1:
                output_img = output_img.squeeze(0)
            if target_img.shape[0] == 1:
                target_img = output_img.squeeze(0)
            
            # Normalize to [0,1] for saving
            input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min() + 1e-8)
            output_img = (output_img - output_img.min()) / (output_img.max() - output_img.min() + 1e-8)
            target_img = (target_img - target_img.min()) / (target_img.max() - target_img.min() + 1e-8)
            
            input_images.append(input_img)
            output_images.append(output_img)
            target_images.append(target_img)
        if len(input_images) >= n_grid:
            break

    # Stack and make grid
    input_grid = vutils.make_grid([img.unsqueeze(0) if img.ndim==2 else img for img in input_images], nrow=4, padding=2, normalize=False)
    output_grid = vutils.make_grid([img.unsqueeze(0) if img.ndim==2 else img for img in output_images], nrow=4, padding=2, normalize=False)
    target_grid = vutils.make_grid([img.unsqueeze(0) if img.ndim==2 else img for img in target_images], nrow=4, padding=2, normalize=False)
    

    save_dir = os.path.dirname(f"{file}_{model._get_name()}_input_grid.png")
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    to_pil(input_grid).save(f"{file}_{model._get_name()}_input_grid.png")
    to_pil(output_grid).save(f"{file}_{model._get_name()}_output_grid.png")
    to_pil(target_grid).save(f"{file}_{model._get_name()}_target_grid.png")

