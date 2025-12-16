from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

DATA_PATH = "/opt/airflow/data/train.csv"
MODEL_PATH = "/opt/airflow/models/model.joblib"

def load_data():
    df = pd.DataFrame({
        "x": [1, 2, 3, 4, 5],
        "y": [2, 4, 6, 8, 10]
    })
    os.makedirs("/opt/airflow/data", exist_ok=True)
    df.to_csv(DATA_PATH, index=False)

def train_model():
    df = pd.read_csv(DATA_PATH)
    X = df[["x"]]
    y = df["y"]
    model = LinearRegression()
    model.fit(X, y)
    os.makedirs("/opt/airflow/models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

def evaluate_model():
    model = joblib.load(MODEL_PATH)
    prediction = model.predict([[6]])
    print("Prediction for x=6:", prediction)

with DAG(
    dag_id="mlops_pipeline",
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False
) as dag:


    load = PythonOperator(
        task_id="load_data",
        python_callable=load_data
    )

    train = PythonOperator(
        task_id="train_model",
        python_callable=train_model
    )

    evaluate = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model
    )

    load >> train >> evaluate
