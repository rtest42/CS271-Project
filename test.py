from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, log_loss, mean_absolute_error, root_mean_squared_error, precision_recall_curve, r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

## Load CSV and preprocess data
df = pd.read_csv("USW00023293.csv", na_values=["", " ", "NA", "N/A", "null", "NULL", "NaN", "nan", "-9999"])

df["DATE"] = pd.to_datetime(df["DATE"])
df = df.sort_values("DATE").set_index("DATE")

cols = ["TMIN", "TMAX", "TAVG"]
df = df[cols]

df = df.interpolate(method="time")
df = df.fillna(df.mean(numeric_only=True))

## Time-based split
split_idx = int(len(df) * 0.8)

train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

## Feature engineering
# Seasonal baseline (computed ONLY on train)
monthly_mean = train_df.groupby(train_df.index.month)["TMAX"].mean() # type: ignore

train_df["TMAX_ANOMALY"] = train_df["TMAX"] - train_df.index.month.map(monthly_mean) # type: ignore
test_df["TMAX_ANOMALY"] = test_df["TMAX"] - test_df.index.month.map(monthly_mean) # type: ignore


# Seasonal threshold (95th percentile, train only)
monthly_95 = train_df.groupby(train_df.index.month)["TMAX"].quantile(0.95) # type: ignore

train_df["HEAT_CLASS"] = (train_df["TMAX"] >= train_df.index.month.map(monthly_95)).astype(int) # type: ignore
test_df["HEAT_CLASS"] = (test_df["TMAX"] >= test_df.index.month.map(monthly_95)).astype(int) # type: ignore


# Add seasonality encoding
for d in [train_df, test_df]:
    d["month_sin"] = np.sin(2 * np.pi * d.index.month / 12) # type: ignore
    d["month_cos"] = np.cos(2 * np.pi * d.index.month / 12) # type: ignore

## Feature and target selection
features = ["TMIN", "TMAX", "TAVG", "TMAX_ANOMALY", "month_sin", "month_cos"]

X_train = train_df[features]
y_train = train_df["HEAT_CLASS"]

X_test = test_df[features]
y_test = test_df["HEAT_CLASS"]

## Model and evaluation (random forest with threshold tuning)
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

# Threshold tuning
probs = model.predict_proba(X_test)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_test, probs)

# Choose best F1 threshold
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
best_idx = np.argmax(f1_scores)

best_threshold = thresholds[max(best_idx - 1, 0)]

pred = (probs >= best_threshold).astype(int)

# Evaluation
print("Best threshold:", best_threshold)
print("\nClass distribution (train):")
print(y_train.value_counts(), "\n")

print("Class distribution (test):")
print(y_test.value_counts(), "\n")
print(classification_report(y_test, pred))

## ----------------------------

## Lag features
df["TMAX_lag1"] = df["TMAX"].shift(1)
df["TMAX_lag2"] = df["TMAX"].shift(2)
df["TAVG_lag1"] = df["TAVG"].shift(1)

## Regression target
df["TMAX_next"] = df["TMAX"].shift(-1)
df = df.dropna()

## Features and target
features = ["TMAX", "TAVG", "TMIN", "TMAX_lag1", "TMAX_lag2", "TAVG_lag1"]

X = df[features]
y = df["TMAX_next"]

# Train test split
split = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

## Model and evaluatinon (MLP)
mlp = Pipeline([
    ("scaler", StandardScaler()),
    ("model", MLPRegressor(
        hidden_layer_sizes=(64, 32),
        max_iter=3000,
        learning_rate_init=0.001,
        random_state=42
    ))
])

mlp.fit(X_train, y_train)
pred = mlp.predict(X_test)

pred_train = mlp.predict(X_train)
pred_test = mlp.predict(X_test)

# MLP Stats
print("Train MAE:", mean_absolute_error(y_train, pred_train))
print("Test MAE:", mean_absolute_error(y_test, pred_test))

print("Train RMSE:", root_mean_squared_error(y_train, pred_train))
print("Test RMSE:", root_mean_squared_error(y_test, pred_test))

print("Test R²:", r2_score(y_test, pred_test))

# MLP Graph
plt.plot(mlp.named_steps["model"].loss_curve_)
plt.title("MLP Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()