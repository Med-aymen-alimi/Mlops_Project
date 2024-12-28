from operator import index
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sklearn
from fastapi import FastAPI, File, UploadFile, Request
import uvicorn
import sys  
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import mlflow
from src.clean_data_csv import clean_data
from src.clean_data_json import clean_data_json
from example_json.transaction_info import TransactionModel
import os
import mlflow.pyfunc
from dotenv import load_dotenv

load_dotenv(r"C:/work/Studies/Finalyear/1stsemester/Bassem/repo_clone/MLOps-Training/backend/src/.env")

# Fetch environment variables (without dotenv)
DagsHub_username = os.getenv("DagsHub_username")
DagsHub_token = os.getenv("DagsHub_token")

# Ensure that credentials are loaded properly
if DagsHub_username is None or DagsHub_token is None:
    raise ValueError("DagsHub username or token not found in environment variables.")

# Setup MLflow
mlflow.set_tracking_uri('https://dagshub.com/aymenalimii4070/Ml_OPS_Movies.mlflow')

# Set up FastAPI application
app = FastAPI()
origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Query all experiments and runs in MLflow
all_experiments = [exp.experiment_id for exp in mlflow.search_experiments()]
df_mlflow = mlflow.search_runs(experiment_ids=all_experiments)

# Find the best run based on the desired metric (F1_score_test)
metric_column = 'metrics.F1_score_test'
if metric_column in df_mlflow.columns:
    best_run = df_mlflow.loc[df_mlflow[metric_column].idxmax()]  # Maximum F1 score
    run_id = best_run['run_id']
    print(f"Selected run ID: {run_id}")
else:
    raise KeyError(f"Metric '{metric_column}' not found in the dataframe. Available columns: {df_mlflow.columns}")

# Load the model using the best run ID
logged_model = f'runs:/{run_id}/ML_models'
model = mlflow.pyfunc.load_model(logged_model)

@app.get("/")
def read_root():
    return {"Hello": "to fraud detector app version 2"}

# This endpoint receives data in the form of CSV (historical transactions data)
@app.post("/predict/csv")
def return_predictions(file: UploadFile = File(...)):
    data = pd.read_csv(file.file)
    preprocessed_data = clean_data(data)  # Preprocessing the data
    predictions = model.predict(preprocessed_data)  # Predicting using the model
    return {"predictions": predictions.tolist()}

# This endpoint receives data in the form of JSON (information about one transaction)
@app.post("/predict")
async def predict(data: TransactionModel):
    try:
        data_dict = data.dict()
        preprocessed_data = clean_data_json(data_dict)  # Preprocessing the data from JSON
        print(preprocessed_data)
        predictions = model.predict(preprocessed_data)  # Predicting using the model
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8080)