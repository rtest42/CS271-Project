import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from conv2d_rnnmodels import Conv2dLSTM

class ClimateDataset(Dataset):
    def __init__(self, data, window=24):
        self.data = data
        self.window = window

    def __len__(self):
        return len(self.data) - self.window

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.window].unsqueeze(1)  # shape: (window, 1, lat, lon)
        y = self.data[idx+self.window].unsqueeze(0)  # shape: (1, lat, lon)

        return x, y

ds = xr.open_dataset("2m_temperature_5.625deg/2m_temperature_1999_5.625deg.nc")

data = torch.tensor(ds["t2m"].values - 273.15, dtype=torch.float32)  # convert from Kelvin to Celsius
dataset = ClimateDataset(data, window=24)
split_idx = int(0.8 * len(dataset))

train_dataset = torch.utils.data.Subset(
    dataset,
    range(split_idx)
)

test_dataset = torch.utils.data.Subset(
    dataset,
    range(split_idx, len(dataset))
)
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    # persistent_workers=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    num_workers=0,
    pin_memory=True,
    # persistent_workers=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Conv2dLSTM(
    input_size=1,
    hidden_size=16,
    kernel_size=3,
    num_layers=1,
    bias=True,
    output_size=1
    ).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    patience=2,
    factor=0.5
)
num_epochs = 10
scaler = torch.amp.GradScaler("cuda")
for epoch in tqdm(range(num_epochs)):

    model.train()
    total_loss = 0

    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
            
        with torch.amp.autocast("cuda"):
            pred = model(x)
            loss = criterion(pred, y)
        
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    scheduler.step(avg_loss)


    print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.6f}")

model.eval()

test_loss = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)

        with torch.amp.autocast("cuda"):
            pred = model(x)
            loss = criterion(pred, y)

        test_loss += loss.item()

avg_test_loss = test_loss / len(test_loader)
print(f"\nTest Loss: {avg_test_loss:.6f}")

x, y = dataset[split_idx]  # Get the first sample from the test set
with torch.no_grad():
    pred = model(x.unsqueeze(0).to(device)).cpu().squeeze()

true = y.squeeze().numpy()
error = pred - true

fig, axes = plt.subplots(1, 3, figsize=(18,5))

vmin, vmax = -30, 40

# --- Actual ---
im0 = axes[0].imshow(true, cmap="nipy_spectral", vmin=vmin, vmax=vmax, origin="lower", extent=[float(ds.lon.min() - 180), float(ds.lon.max() - 180), float(ds.lat.min()), float(ds.lat.max())])
axes[0].set_title("Actual Temperature")
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

# --- Predicted ---
im1 = axes[1].imshow(pred, cmap="nipy_spectral", vmin=vmin, vmax=vmax, origin="lower", extent=[float(ds.lon.min() - 180), float(ds.lon.max() - 180), float(ds.lat.min()), float(ds.lat.max())])
axes[1].set_title("Predicted Temperature")
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

# --- Error (Difference) ---
im2 = axes[2].imshow(error, cmap="bwr", vmin=-3, vmax=3, origin="lower", extent=[float(ds.lon.min() - 180), float(ds.lon.max() - 180), float(ds.lat.min()), float(ds.lat.max())])
axes[2].set_title("Prediction Error (Pred - Actual)")
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()
