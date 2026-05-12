import torch
from torch.utils.data import Dataset
import numpy as np

class ClimateDataset(Dataset):
    def __init__(self, data, seq_len=7):
        """
        data: numpy array of shape (num_days, 3) -> [TMIN, TMAX, TAVG]
        """
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]          # (seq_len, 3)
        y = self.data[idx+self.seq_len, 2]           # predict TAVG only

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return x, y
    
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=64):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, features)

        out, _ = self.lstm(x)

        out = out[:, -1, :]   # last timestep only
        out = self.fc(out)    # (batch, 1)

        return out
    
def train(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        # FIX SHAPES HERE
        y = y.view(-1, 1)   # (batch, 1)

        preds = model(x)    # (batch, 1)

        loss = criterion(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0
    preds_list = []
    true_list = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            y = y.view(-1, 1)

            preds = model(x)

            loss = criterion(preds, y)
            total_loss += loss.item()

            # FIX: safe conversion
            preds_list.extend(preds.cpu().view(-1).numpy())
            true_list.extend(y.cpu().view(-1).numpy())

    return total_loss / len(loader), preds_list, true_list

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTMModel().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)