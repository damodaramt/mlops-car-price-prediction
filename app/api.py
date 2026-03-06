from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd

app = FastAPI()

# Tell API where MLflow server is
mlflow.set_tracking_uri("http://host.docker.internal:5000")

# Load production model
model = mlflow.pyfunc.load_model("models:/car_price_model/Production")


@app.get("/")
def home():
    return {"message": "Car Price Prediction API Running"}


@app.post("/predict")
def predict(curbweight: float, enginesize: float, horsepower: float):

    data = pd.DataFrame([[curbweight, enginesize, horsepower]],
                        columns=["curbweight","enginesize","horsepower"])

    prediction = model.predict(data)

    return {"predicted_price": float(prediction[0])}
