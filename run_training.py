import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from diffnet import DiffNet
from unet import UNet  # Assumes UNet is defined in unet.py
from load_dataset import load_dataset  # Assumes this returns a torch Dataset

# Hyperparameters
batch_size = 1
epochs = 100
learning_rate = 1e-3
patience = 10  # Early stopping patience
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = load_dataset()
n_total = len(dataset)
n_val = n_total // 10
n_test = n_total // 10
n_train = n_total - n_val - n_test

train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size)
test_loader = DataLoader(test_set, batch_size=batch_size)

# Model, loss, optimizer

for model in [[UNet(n_channels=1, n_classes=1), DiffNet(in_channels=1, out_channels=1)]]:
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    def evaluate(loader):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                total_loss += loss.item() * x.size(0)
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
        val_loss = evaluate(val_loader)
        print(f"Epoch {epoch+1}/{epochs} - Validation Loss: {val_loss:.4f}")
    
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
    # Load best model and evaluate on test set
    model.load_state_dict(torch.load("best_model.pt"))
    test_loss = evaluate(test_loader)
    print(f"Test Loss: {test_loss:.4f}")