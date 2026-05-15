import os
import xarray as xr
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from dataclasses import dataclass, field
from glob import glob
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
from typing import cast

# =========================================================
# Data Structures
# =========================================================

@dataclass
class DatasetSplit:
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    ds: xr.Dataset
    grid_coords: xr.DataArray


@dataclass
class EvaluationResult:
    predictions: np.ndarray = field(repr=False)
    mae: float
    rmse: float
    r2: float


# =========================================================
# Model
# =========================================================

class KernelRidgeTempPredictor:
    """Handles training, prediction, and evaluation of the Kernel Ridge model."""
    def __init__(self, alpha: float = 1.0, gamma: float = 0.01):
        self.alpha = alpha
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95)),
            ("krr", KernelRidge(
                alpha=alpha,
                kernel="rbf",
                gamma=gamma
            ))
        ])

    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)

    def evaluate(self, x: np.ndarray, x_flat: np.ndarray, y: np.ndarray) -> EvaluationResult:
        base = persistence(x)
        preds = base + self.model.predict(x_flat)

        return EvaluationResult(
            predictions=preds,
            mae=cast(float, mean_absolute_error(y, preds)),
            rmse=root_mean_squared_error(y, preds),
            r2=r2_score(y, preds)
        )

# =========================================================
# Data Preparation
# =========================================================

def prepare_data(filepaths: list[str] | str, window: int = 24, train_split: float = 0.7) -> DatasetSplit:
    """Loads NetCDF data and creates sliding window features."""
    # TODO: Handle more data (breaks after two 72KB files)
    if isinstance(filepaths, list):
        ds = xr.open_mfdataset(filepaths, combine="by_coords", chunks={"time": window})
    else:
        ds = xr.open_dataset(filepaths)

    # Convert Kelvin to Celsius
    temp = cast(xr.DataArray, cast(object, ds["t2m"] - 273.15))
    temp = temp.stack(grid=("lat", "lon"))
    temp_np = np.asarray(temp.values, dtype=np.float32)

    # Sliding windows
    x = np.lib.stride_tricks.sliding_window_view(
        temp_np, window_shape=window, axis=0
    )[:-1]  # shape: (samples, grid, window)
    y = temp_np[window:window + len(x)]

    # Reshape
    x = x.transpose(0, 2, 1)

    # split (time-ordered)
    train_end = int(train_split * len(x))
    val_end = int((train_split + (1 - train_split) / 2) * len(x))
    split = lambda arr: (arr[:train_end], arr[train_end:val_end], arr[val_end:])

    return DatasetSplit(
        *split(x),
        *split(y),
        ds=ds,
        grid_coords=temp.grid
    )


# =========================================================
# Visualization
# =========================================================

def to_spatial_map(values: np.ndarray, grid_coords: xr.DataArray) -> xr.DataArray:
    """Converts flattened grid predictions back into (lat, lon) format using xarray unstack()."""
    return xr.DataArray(
        values,
        dims=["grid"],
        coords={"grid": grid_coords}
    ).unstack("grid")


def plot_map(ax, data__, title, cmap, clim, extent):
    """Helper function for plotting."""
    im = ax.imshow(data__, cmap=cmap, vmin=clim[0], vmax=clim[1], origin="lower", extent=extent)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def visualize_prediction(y_true: np.ndarray, y_pred: np.ndarray, grid_coords: xr.DataArray, ds: xr.Dataset,
                         idx: int = 0, save_path: str = "prediction_map.png") -> None:
    """Plots the actual, predicted, and error maps."""
    true_map = to_spatial_map(y_true[idx], grid_coords)
    pred_map = to_spatial_map(y_pred[idx], grid_coords)
    error_map = pred_map - true_map

    extent = [
        float(ds.lon.min() - 180),
        float(ds.lon.max() - 180),
        float(ds.lat.min()),
        float(ds.lat.max())
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    maps = [
        (true_map, "Actual Temperature", "nipy_spectral", (-20, 40)),
        (pred_map, "Predicted Temperature", "nipy_spectral", (-20, 40)),
        (error_map, "Prediction Error", "bwr", (None, None))
    ]

    for ax, (data_, title, cmap, clim) in zip(axes, maps):
        plot_map(ax=ax, data__=data_, title=title, cmap=cmap, clim=clim, extent=extent)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    mlflow.log_artifact(save_path, artifact_path="plots")


# =========================================================
# Baseline
# =========================================================

def persistence(x):
    return x[:, -1, :]


def main(file_paths: list[str] | str, window_values: list[int], alpha_values: list[float], gamma_values: list[float]):
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    mlflow.set_experiment("kernel_ridge_residual")

    param_dist = {
        "window_size": window_values,
        "alpha": alpha_values,
        "gamma": gamma_values
    }

    param_samples = list(ParameterGrid(param_dist))

    data = prepare_data(filepaths=file_paths, window=max(window_values))

    best_val_rmse = float("inf")
    best_params = None
    with mlflow.start_run(run_name="kernel_ridge_sweep"):
        for params in tqdm(param_samples, position=0):
            window_ = params["window_size"]
            alpha_ = params["alpha"]
            gamma_ = params["gamma"]

            # Feature slicing
            x_train = data.x_train[:, -window_:, :]
            x_val = data.x_val[:, -window_:, :]
            x_test = data.x_test[:, -window_:, :]

            # Flatten ONLY for model
            x_train_flat = x_train.reshape(len(x_train), -1)
            x_val_flat = x_val.reshape(len(x_val), -1)
            x_test_flat = x_test.reshape(len(x_test), -1)

            gamma_effective = gamma_ / x_train_flat.shape[1]

            # Residual target
            data_window = data
            y_train_res = data.y_train - persistence(x_train)

            # Train model
            model = KernelRidgeTempPredictor(alpha=alpha_, gamma=gamma_effective)
            model.train(x_train_flat, y_train_res)

            # Evaluate
            train_result = model.evaluate(x_train, x_train_flat, data_window.y_train)
            val_result = model.evaluate(x_val, x_val_flat, data_window.y_val)
            test_result = model.evaluate(x_test, x_test_flat, data_window.y_test)

            # Persistence baseline
            baseline_rmse = root_mean_squared_error(data_window.y_test, persistence(x_test))

            with mlflow.start_run(run_name=f"w{window_}_a{alpha_}_g{gamma_}", nested=True):
                # Log parameters
                mlflow.log_param("window_size", window_)
                mlflow.log_param("alpha", alpha_)
                mlflow.log_param("gamma", gamma_)
                mlflow.log_param("gamma_effective", gamma_effective)

                # Log metrics
                improvement = (baseline_rmse - test_result.rmse) / baseline_rmse * 100
                mlflow.log_metric("train_mae", train_result.mae)
                mlflow.log_metric("train_rmse", train_result.rmse)
                mlflow.log_metric("train_r2", train_result.r2)
                mlflow.log_metric("val_mae", val_result.mae)
                mlflow.log_metric("val_rmse", val_result.rmse)
                mlflow.log_metric("val_r2", val_result.r2)
                mlflow.log_metric("test_mae", test_result.mae)
                mlflow.log_metric("test_rmse", test_result.rmse)
                mlflow.log_metric("test_r2", test_result.r2)
                mlflow.log_metric("baseline_rmse", baseline_rmse)
                mlflow.log_metric("rmse_improvement", improvement)

                # Save the best model
                if val_result.rmse < best_val_rmse:
                    best_val_rmse = val_result.rmse
                    best_params = params

                    # Log model
                    mlflow.sklearn.log_model(
                        sk_model=model.model,
                        name=f"kernel_ridge_w{window_}_a{str(alpha_).replace('.', '_')}_g{str(gamma_).replace('.', '_')}",
                        serialization_format="skops",
                        pip_requirements=["scikit-learn", "numpy"]
                    )

                # Print results
                tqdm.write(f"Parameters for kernel ridge: window_size={window_} hours, alpha={alpha_}, gamma={gamma_}")
                tqdm.write(f"Test Metrics: {str(test_result)}")
                tqdm.write(f"Baseline improvement: {improvement:.2f}%")

                # Visualize using plots
                visualize_prediction(
                    y_true=data_window.y_test,
                    y_pred=test_result.predictions,
                    grid_coords=data_window.grid_coords,
                    ds=data_window.ds,
                    idx=0,
                    save_path=f"{output_dir}/kernel_ridge_w{window_}_a{str(alpha_).replace('.', '_')}_g{str(gamma_).replace('.', '_')}.png"
                )

        tqdm.write(f"Best Parameters: {str(best_params)}")
        tqdm.write(f"Best validation RMSE: {best_val_rmse:.4f}")

if __name__ == "__main__":
    # Config
    file_paths_ = "2m_temperature_5.625deg/2m_temperature_2005_5.625deg.nc" # sorted(glob("2m_temperature_5.625deg/2m_temperature_????_5.625deg.nc"))
    window_values_ = [24] # [6, 12, 24, 48, 72, 168, 720]
    alpha_values_ = [1.0] # list(np.logspace(-2, 2, 10))
    gamma_values_ = [1e-2]
    # alpha_values_ = [10.0] # [0.01, 0.1, 1.0, 10.0, 100.0]

    main(file_paths=file_paths_, window_values=window_values_, alpha_values=alpha_values_, gamma_values=gamma_values_)
