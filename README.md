# MLOps Car Price Prediction

End-to-end MLOps pipeline for predicting car prices using Scikit-Learn, MLflow, FastAPI, and Docker.

## 🚀 Project Architecture

Dataset → Training Pipeline → MLflow Tracking → Model Registry → FastAPI → Docker API

## 🐳 Docker Deployment

Build the Docker image

docker build -t car-price-api .

Run the container

docker run -p 8000:8000 car-price-api

## 📡 API Example

POST /predict

Example:

http://SERVER_IP:8000/predict?curbweight=2548&enginesize=130&horsepower=111

Response:

{
 "predicted_price": 13805
}

## 📦 Tech Stack
- Python
- Scikit-Learn
- MLflow
- FastAPI
- Docker
- GitHub

## 📊 Training Pipeline
The model is trained using the following features:
- curbweight
- enginesize
- horsepower

Target:
- price

## ⚙️ Run Training

## 📂 Project Structure

mlops-car-price-prediction
│
├── app
│   └── api.py
│
├── src
│   └── train.py
│
├── data
│
├── Dockerfile
├── requirements.txt
└── .gitignore

## 👨‍💻 Author

Damodaram T  
MLOps Engineer Project


