import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow
import numpy as np
import os
import mlflow.sklearn

# load dataset
df = pd.read_csv("data/CarPrice_Assignment.csv")

# features
X = df[["curbweight","enginesize","horsepower"]]

# target
y = df["price"]

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train model
model = LinearRegression()
model.fit(X_train, y_train)

# predictions
preds = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))

# mlflow tracking
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(tracking_uri)

mlflow.set_experiment("car-price-experiment")

with mlflow.start_run():

    mlflow.log_metric("rmse", rmse)

    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name="car_price_model"
    )

print("Training completed")
