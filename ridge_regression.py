import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

ds = xr.open_mfdataset("2m_temperature_5.625deg/2m_temperature_1999_5.625deg.nc", combine="by_coords")

data = ds["t2m"].values - 273.15  # convert from Kelvin to Celsius
time = ds.time.values
time_len, lat, lon = data.shape

window = 24  # hours

X, y, y_times = [], [], []

for t in range(window, time_len - 1):
    X.append(data[t-window:t].reshape(-1))  # flatten spatial dimensions
    y.append(data[t].reshape(-1))  # predict all grid points
    y_times.append(time[t])  # store the corresponding time steps

X = np.array(X)  # (num_samples, window * lat * lon)
y = np.array(y)  # (num_samples, lat * lon) 
y_times = np.array(y_times)  # (num_samples)

# Train-test split
split = int(len(X) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# --- Train Ridge Regression ---
model = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1.0, random_state=42))
])

model.fit(X_train, y_train)

pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

## Stats
print("Train MAE:", mean_absolute_error(y_train, pred_train))
print("Test MAE:", mean_absolute_error(y_test, pred_test))

print("Train RMSE:", root_mean_squared_error(y_train, pred_train))
print("Test RMSE:", root_mean_squared_error(y_test, pred_test))

print("Test R²:", r2_score(y_test, pred_test))

i = 0  # index of the time step to visualize
true = np.array(y_test[i]).reshape(lat, lon)
pred = np.array(pred_test[i]).reshape(lat, lon)
error = pred - true

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --- Actual ---
im0 = axes[0].imshow(true, cmap="nipy_spectral", vmin=-20, vmax=40, origin="lower", extent=[float(ds.lon.min() - 180), float(ds.lon.max() - 180), float(ds.lat.min()), float(ds.lat.max())])
axes[0].set_title("Actual Temperature")
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

# --- Predicted ---
im1 = axes[1].imshow(pred, cmap="nipy_spectral", vmin=-20, vmax=40,origin="lower", extent=[float(ds.lon.min() - 180), float(ds.lon.max() - 180), float(ds.lat.min()), float(ds.lat.max())])
axes[1].set_title("Predicted Temperature")
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

# --- Error (Difference) ---
im2 = axes[2].imshow(error, cmap="bwr", vmin=-3, vmax=3, origin="lower", extent=[float(ds.lon.min() - 180), float(ds.lon.max() - 180), float(ds.lat.min()), float(ds.lat.max())])
axes[2].set_title("Prediction Error (Pred - Actual)")
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()