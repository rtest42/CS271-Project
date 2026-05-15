import itertools

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

    preds = []
    trues = []

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

            preds.append(pred)
            trues.append(y)
            loss = criterion(pred, y)
            total_loss += loss.item()
    preds = torch.cat(preds, dim=0)
    trues = torch.cat(trues, dim=0)

    mse = torch.mean((preds - trues) ** 2)
    rmse = torch.sqrt(mse)

    return mse.item()

# =========================================================
# Main
# =========================================================

def run_experiment(config, train_dataset, val_dataset, test_dataset, dataset, ds, device, use_amp, window, val_end, gap, std, mean):
    hidden_size = config["hidden_size"]
    num_layers = config["num_layers"]
    kernel_size = config["kernel_size"]
    batch_size = config["batch_size"]
    lr = config["lr"]

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=use_amp,
        # persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_amp,
        # persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_amp,
        # persistent_workers=True
    )

    # Convolutional 2D LSTM
    model = Conv2dLSTM(
        input_size=1,
        hidden_size=hidden_size,
        kernel_size=kernel_size,
        num_layers=num_layers,
        bias=True,
        output_size=1
    ).to(device)

    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
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

    with mlflow.start_run(run_name=f"h{hidden_size}_k{kernel_size}_lr{lr}_b{batch_size}", nested=True) as run:
        mlflow.log_param("window", window)
        mlflow.log_params(config)
        best_model_path = "best_model_{}.pt".format(run.info.run_id)
        # Training loop
        for epoch in range(num_epochs):

            model.train()
            train_loss = 0

            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False, disable=True)
            for x, y in loop:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
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
            mlflow.log_metric("lr", optimizer.param_groups[0]["lr"], step=epoch+1)
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

                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    tqdm.write("Early stopping triggered.")
                    break

        # Load best model
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        mlflow.pytorch.log_model(model, name="model", pip_requirements=["torch==2.11.0+cu130", "numpy", "xarray"])
        os.remove(best_model_path)

        # Test evaluation
        avg_test_loss = evaluate(
            model,
            test_loader,
            criterion,
            device,
            use_amp
        )

        rmse = avg_test_loss ** 0.5
        mlflow.log_metric("test_loss", avg_test_loss)
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("best_val_loss", best_val_loss)
        test_rmse_c = rmse * std.item()
        mlflow.log_metric("test_rmse_c", test_rmse_c)
        mlflow.log_metric("best_val_rmse_c", (best_val_loss ** 0.5) * std.item())
        tqdm.write(f"Test Loss: {avg_test_loss:.6f} / Test RMSE: {rmse:.6f} (STD: {std.item():.6f})")
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
        save_path = (
            f"{output_dir}/"
            f"conv2d_lstm_w{window}"
            f"_h{hidden_size}"
            f"_k{kernel_size}"
            f"_b{batch_size}"
            f"_lr{str(lr).replace('.', '_')}.png"
        )
        plt.savefig(save_path)
        plt.close(fig)
        mlflow.log_artifact(save_path, artifact_path="plots")
        return best_val_loss


def main(params: dict, filename: str):
    mlflow.set_experiment("conv2d_lstm")

    keys = list(params.keys())
    configs = list(itertools.product(*params.values()))

    best_score = float("inf")
    best_config = None

    # Load data
    ds = xr.open_dataset(filename)

    data = torch.tensor(ds["t2m"].values - 273.15, dtype=torch.float32)  # convert from Kelvin to Celsius
    mean = data.mean()
    std = data.std()
    data = (data - mean) / std  # normalize data

    # Device assignment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    with mlflow.start_run(run_name="conv2d_lstm_sweep"):
        mlflow.log_param("num_configs", len(configs))
        mlflow.log_param("dataset", os.path.basename(filename))

        for values in tqdm(configs, desc="Hyperparameter tuning"):
            config = dict(zip(keys, values))


            # Dataset
            window = config["window_size"]

            dataset = ClimateDataset(data, window=window)

            gap = window  # Prevent temporal leakage

            n = len(dataset)

            train_end = int(0.7 * n)
            val_end = int(0.85 * n)

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

            tqdm.write(f"Running config {config}")

            score = run_experiment(
                config=config,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                dataset=dataset,
                ds=ds,
                device=device,
                use_amp=use_amp,
                window=window,
                val_end=val_end,
                gap=gap,
                mean=mean,
                std=std,
            )

            if score < best_score:
                best_score = score
                best_config = config

        tqdm.write(f"Best config: {best_config}")
        tqdm.write(f"Best val loss: {best_score}")

        mlflow.log_metric("best_val_loss", best_score)
        if best_config is not None:
            for k, v in best_config.items():
                mlflow.log_param(f"best_{k}", v)


if __name__ == "__main__":
    # Configure hyperparameters here.
    param_grid = {
        "window_size": [24],
        "hidden_size": [16],
        "num_layers": [1],
        "kernel_size": [3],
        "lr": [1e-3],
        "batch_size": [16]
    }
    file_name = "2m_temperature_5.625deg/2m_temperature_2005_5.625deg.nc"

    main(params=param_grid, filename=file_name)
