import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

mlflow.set_tracking_uri("http://localhost:5000")

X, y = load_iris(return_X_y=True)

with mlflow.start_run():

    model = RandomForestClassifier()
    model.fit(X, y)

    mlflow.log_param("model", "RandomForest")
    mlflow.log_metric("accuracy", model.score(X, y))

    mlflow.sklearn.log_model(model, "model")

    print("Run logged successfully")
