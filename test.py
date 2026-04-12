from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, log_loss, mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np

df = pd.read_csv("USW00023293.csv", na_values=["", " ", "NA", "N/A", "null", "NULL", "NaN", "nan", "-9999"])

df["DATE"] = pd.to_datetime(df["DATE"])
df = df.sort_values("DATE")
df = df.set_index("DATE")

cols = ["TMIN", "TMAX", "TAVG"]
df = df[cols]

df = df.interpolate(method="time")
df = df.fillna(df.mean(numeric_only=True))

tmax_95 = df["TMAX"].quantile(0.95)
tmax_99 = df["TMAX"].quantile(0.99)

print("95th percentile:", tmax_95)
print("99th percentile:", tmax_99)

def classify_heat(tmax):
    if tmax >= tmax_95:
        return 1  # Heat Advisory
    else:
        return 0  # Normal

df["HEAT_CLASS"] = df["TMAX"].apply(classify_heat)

df["is_hot_day"] = df["TMAX"] >= tmax_95

# rolling 3-day heatwave
df["heatwave_3d"] = df["is_hot_day"].rolling(3).sum() >= 3

df["HEATWAVE_EVENT"] = df["heatwave_3d"].astype(int)

X = df[["TMIN", "TMAX", "TAVG"]]
X = X.fillna(X.mean())

y = df["HEAT_CLASS"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, shuffle=False
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)

print(y.value_counts())

print(classification_report(y_test, pred))

# exit()

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(
    hidden_layer_sizes=(32, 16), 
    alpha=0.001,
    learning_rate_init=0.001,
    max_iter=500,
    random_state=42
)
mlp.fit(X_train, y_train)

print("Test accuracy:", mlp.score(X_test, y_test))
train_loss = log_loss(y_train, mlp.predict_proba(X_train))
test_loss = log_loss(y_test, mlp.predict_proba(X_test))

print("Train loss:", train_loss)
print("Test loss:", test_loss)
print("MAE:", mean_absolute_error(y_test, pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))

import matplotlib.pyplot as plt

plt.plot(mlp.loss_curve_)
plt.title("MLP Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()