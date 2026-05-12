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
        x = self.data[idx:idx+self.window]
        y = self.data[idx+self.window]

        x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)

        return x, y
    
ds = xr.open_mfdataset("2m_temperature_5.625deg/2m_temperature_1999_5.625deg.nc", combine="by_coords")

data = ds["t2m"].values - 273.15  # convert from Kelvin to Celsius
dataset = ClimateDataset(data, window=24)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4)

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

for epoch in tqdm(range(10)):

    model.train()
    total_loss = 0

    for x, y in train_loader:

        x, y = x.to(device), y.to(device)

        pred = model(x)

        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")

model.eval()

x, y = test_dataset[0]

with torch.no_grad():
    pred = model(x.unsqueeze(0).to(device)).cpu().squeeze().numpy()

true = y.squeeze().numpy()
error = pred - true

fig, axes = plt.subplots(1, 3, figsize=(18,5))

vmin, vmax = -30, 40

axes[0].imshow(true, cmap="coolwarm", vmin=vmin, vmax=vmax)
axes[0].set_title("Actual")

axes[1].imshow(pred, cmap="coolwarm", vmin=vmin, vmax=vmax)
axes[1].set_title("Predicted")

axes[2].imshow(error, cmap="bwr")
axes[2].set_title("Error")

plt.tight_layout()
plt.show()
