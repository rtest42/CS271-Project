import mlflow
import mlflow.pytorch
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import matplotlib.pyplot as plt
from Conv2DRNN.conv2d_rnnmodels import Conv2dLSTM

# =========================================================
# Dataset
# =========================================================

class ClimateDataset(Dataset):
    """Dataset for sliding window of temperature data."""
    def __init__(self, data, window: int = 24):
        self.data = data
        self.window = window

    def __len__(self):
        return len(self.data) - self.window

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.window].unsqueeze(1)  # shape: (window, 1, lat, lon)
        y = self.data[idx+self.window].unsqueeze(0)  # shape: (1, lat, lon)

        return x, y


# =========================================================
# Validation/Test Function
# =========================================================

def evaluate(model, loader, criterion, device, use_amp):

    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.amp.autocast(
                device_type=device.type,
                enabled=use_amp
            ):
                pred = model(x)
                loss = criterion(pred, y)

            total_loss += loss.item()

    return total_loss / len(loader)

# =========================================================
# Main
# =========================================================

def main():
    mlflow.set_experiment("conv2d_lstm_model")

    # Load data
    ds = xr.open_dataset("2m_temperature_5.625deg/2m_temperature_1999_5.625deg.nc")

    data = torch.tensor(ds["t2m"].values - 273.15, dtype=torch.float32)  # convert from Kelvin to Celsius
    mean = data.mean()
    std = data.std()
    data = (data - mean) / std  # normalize data

    # Dataset
    window = 24
    dataset = ClimateDataset(data, window=window)

    n = len(dataset)

    train_end = int(0.7 * n)
    val_end = int(0.85 * n)

    gap = window  # Prevent temporal leakage

    train_dataset = torch.utils.data.Subset(
        dataset,
        range(0, train_end)
    )

    val_dataset = torch.utils.data.Subset(
        dataset,
        range(train_end + gap, val_end)
    )

    test_dataset = torch.utils.data.Subset(
        dataset,
        range(val_end + gap, n)
    )

    # Data loaders

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        # persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        # persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        # persistent_workers=True
    )

    # Device assignment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    # Convolutional 2D LSTM
    model = Conv2dLSTM(
        input_size=1,
        hidden_size=16,
        kernel_size=3,
        num_layers=1,
        bias=True,
        output_size=1
    ).to(device)

    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=2,
        factor=0.5
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    num_epochs = 50
    patience = 5
    patience_counter = 0
    best_val_loss = float("inf")

    with mlflow.start_run():
        mlflow.log_param("window", window)
        mlflow.log_param("hidden_size", 16)
        mlflow.log_param("lr", 1e-3)
        mlflow.log_param("batch_size", 16)
        mlflow.log_param("model", "Conv2dLSTM")
        mlflow.log_param("norm_mean", mean.item())
        mlflow.log_param("norm_std", std.item())

        # Training loop
        for epoch in range(num_epochs):

            model.train()
            train_loss = 0

            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
            for x, y in loop:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    pred = model(x)
                    loss: torch.Tensor = criterion(pred, y)

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0
                )

                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # Validation
            avg_val_loss = evaluate(
                model,
                val_loader,
                criterion,
                device,
                use_amp
            )

            scheduler.step(avg_val_loss)

            mlflow.log_metric("train_loss", avg_train_loss, step=epoch+1)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch+1)
            tqdm.write(
                f"Epoch {epoch + 1:02d} | "
                f"Train Loss: {avg_train_loss:.6f} | "
                f"Val Loss: {avg_val_loss:.6f}"
            )

            # Save the best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0

                torch.save(model.state_dict(), "best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    tqdm.write("Early stopping triggered.")
                    break

        # Load best model
        model.load_state_dict(torch.load("best_model.pt"))
        mlflow.pytorch.log_model(model, "model")

        # Test evaluation
        avg_test_loss = evaluate(
            model,
            test_loader,
            criterion,
            device,
            use_amp
        )

        test_rmse = avg_test_loss ** 0.5
        mlflow.log_metric("test_loss", avg_test_loss)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("best_val_loss", best_val_loss)
        tqdm.write(f"Test Loss: {avg_test_loss:.6f} / Test RMSE: {test_rmse:.6f}")
        tqdm.write(f"Best val loss: {best_val_loss:.6f}")

        # Visualization
        model.eval()

        x, y = dataset[val_end + gap]

        with torch.no_grad():
            pred = model(x.unsqueeze(0).to(device)).cpu().squeeze()

        # Denormalize
        pred = pred * std + mean
        true = y.squeeze() * std + mean
        pred = pred.numpy()
        true = true.numpy()
        error = pred - true

        fig, axes = plt.subplots(1, 3, figsize=(18,5))

        extent = [float(ds.lon.min() - 180), float(ds.lon.max() - 180), float(ds.lat.min()), float(ds.lat.max())]

        # Actual temperature graph
        im0 = axes[0].imshow(true, cmap="nipy_spectral", origin="lower", extent=extent)
        axes[0].set_title("Actual Temperature")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        # Predicted temperature graph
        im1 = axes[1].imshow(pred, cmap="nipy_spectral", origin="lower", extent=extent)
        axes[1].set_title("Predicted Temperature")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # Error/difference in temperature graph
        im2 = axes[2].imshow(error, cmap="bwr", origin="lower", extent=extent)
        axes[2].set_title("Prediction Error (Pred - Actual)")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()

        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        save_path = f"{output_dir}/conv2d_lstm_w{window}_hidden{hidden_size}_lr{str(learning_rate).replace('.', '_')}_batch{batch_size}.png"
        plt.savefig(save_path)
        plt.close(fig)
        mlflow.log_artifact(save_path)


if __name__ == "__main__":
    # Configure hyperparameters here.
    param_grid = {
        "hidden_size": [8, 16, 32],
        "num_layers": [1, 2],
        "kernel_size": [3, 5],
        "lr": [1e-3, 3e-4]
    }

    main(params=param_grid)
