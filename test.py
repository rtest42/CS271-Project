from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, log_loss, mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np

df = pd.read_csv("USW00023293.csv", na_values=["", " ", "NA", "N/A", "null", "NULL", "NaN", "nan", "-9999"])

df["DATE"] = pd.to_datetime(df["DATE"])
df = df.sort_values("DATE").set_index("DATE")

cols = ["TMIN", "TMAX", "TAVG"]
df = df[cols]

df = df.interpolate(method="time")
df = df.fillna(df.mean(numeric_only=True))

monthly_mean = df.groupby(df.index.month)["TMAX"].transform("mean") # type: ignore
df["TMAX_ANOMALY"] = df["TMAX"] - monthly_mean

# Seasonal threshold (95th percentile per month)
monthly_95 = df.groupby(df.index.month)["TMAX"].transform(lambda x: x.quantile(0.95)) # type: ignore
df["HEAT_CLASS"] = (df["TMAX"] >= monthly_95).astype(int)

features = ["TMIN", "TMAX", "TAVG", "TMAX_ANOMALY"]
X = df[features]
y = df["HEAT_CLASS"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, shuffle=False, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:, 1]
pred = (probs > 0.33).astype(int)

print("Class distribution:")
print(y.value_counts(), "\n")

print(classification_report(y_test, pred))

exit()

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