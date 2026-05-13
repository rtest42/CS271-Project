import xarray as xr
import matplotlib.pyplot as plt
import mlflow.sklearn
import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from typing import cast

# =========================================================
# Data Structures
# =========================================================

@dataclass
class DatasetSplit:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    ds: xr.Dataset
    grid_coords: xr.DataArray


@dataclass
class EvaluationResult:
    predictions: np.ndarray
    mae: float
    rmse: float
    r2: float

# =========================================================
# Model
# =========================================================

class RidgeRegressionTempPredictor:
    """Handles training, prediction, and evaluation of the Ridge regression model."""
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha))
        ])

    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> EvaluationResult:
        preds = self.predict(x)
        return EvaluationResult(
            predictions=preds,
            mae=cast(float, mean_absolute_error(y, preds)),
            rmse=root_mean_squared_error(y, preds),
            r2=r2_score(y, preds)
        )

# =========================================================
# Data Preparation
# =========================================================

def prepare_data(filepath: str, window: int=24, train_split: float=0.8) -> DatasetSplit:
    """Loads NetCDF data and creates sliding window features."""
    ds = xr.open_dataset(filepath)

    # Convert Kelvin to Celsius
    temp = cast(xr.DataArray, cast(object, ds["t2m"] - 273.15))
    temp = temp.stack(grid=("lat", "lon"))

    time_len = temp.sizes["time"]
    x, y = [], []

    for t in range(window, time_len):
        # Previous timestamps -> features
        x_sample = temp.isel(time=slice(t - window, t)).values.flatten()

        # Current timestamp -> target
        y_sample = temp.isel(time=t).values

        x.append(x_sample)
        y.append(y_sample)

    x, y = np.array(x), np.array(y)
    split_idx = int(len(x) * train_split)

    return DatasetSplit(
        x_train=x[:split_idx],
        x_test=x[split_idx:],
        y_train=y[:split_idx],
        y_test=y[split_idx:],
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

def plot_map(ax, data, title, cmap, clim, extent):
    """Helper function for plotting."""

    im = ax.imshow(
        data,
        cmap=cmap,
        vmin=clim[0],
        vmax=clim[1],
        origin="lower",
        extent=extent
    )

    ax.set_title(title)

    plt.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04
    )

def visualize_prediction(y_true: np.ndarray, y_pred: np.ndarray, grid_coords: xr.DataArray, ds: xr.Dataset, idx: int=0) -> None:
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
        (error_map, "Prediction Error", "bwr", (-3, 3))
    ]

    for ax, (data_, title, cmap, clim) in zip(axes, maps):
        plot_map(ax=ax, data=data_, title=title, cmap=cmap, clim=clim, extent=extent)

    plt.tight_layout()
    plt.savefig("prediction_map.png")
    mlflow.log_artifact("prediction_map.png")
    plt.show()

if __name__ == "__main__":
    file_path = "2m_temperature_5.625deg/2m_temperature_1999_5.625deg.nc"
    window_size = 24
    a = 1.0

    mlflow.set_experiment("ridge_regression")

    data = prepare_data(filepath=file_path, window=window_size)

    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("window_size", window_size)
        mlflow.log_param("alpha", a)
        mlflow.log_param("dataset", file_path)

        # Train model
        predictor = RidgeRegressionTempPredictor(alpha=a)
        predictor.train(data.x_train, data.y_train)

        # Evaluate
        train_result = predictor.evaluate(data.x_train, data.y_train)
        test_result = predictor.evaluate(data.x_test, data.y_test)

        # Log metrics
        mlflow.log_metric("train_mae", train_result.mae)
        mlflow.log_metric("train_rmse", train_result.rmse)
        mlflow.log_metric("train_r2", train_result.r2)
        mlflow.log_metric("test_mae", test_result.mae)
        mlflow.log_metric("test_rmse", test_result.rmse)
        mlflow.log_metric("test_r2", test_result.r2)

        # Log model
        mlflow.sklearn.log_model(
            sk_model=predictor.model,
            name="ridge_model",
            serialization_format="skops"
        )

        # Print results
        print("\n=== Train Metrics ===")
        print(train_result)

        print("\n=== Test Metrics ===")
        print(test_result)

        # Visualize using plots
        visualize_prediction(
            y_true=data.y_test,
            y_pred=test_result.predictions,
            grid_coords=data.grid_coords,
            ds=data.ds,
            idx=0
        )
