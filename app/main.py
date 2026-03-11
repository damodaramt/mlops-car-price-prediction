from fastapi import FastAPI
import mlflow
import mlflow.pyfunc
import pandas as pd
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()

mlflow.set_tracking_uri("http://100.48.130.172:5000")

model = mlflow.pyfunc.load_model("models:/car_price_model/Production")

# Enable Prometheus metrics
Instrumentator().instrument(app).expose(app)


@app.get("/")
def home():
    return {"message": "Car Price Prediction API Running"}


@app.post("/predict")
def predict(curbweight: float, enginesize: float, horsepower: float):

    data = pd.DataFrame(
        [[curbweight, enginesize, horsepower]],
        columns=["curbweight", "enginesize", "horsepower"]
    )

    prediction = model.predict(data)

    return {"predicted_price": float(prediction[0])}
