import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
import mlflow


# mlflow.set_tracking_uri("http://mlflow:5000")
print("Train.py içindeyim.")

def eval_metrics(y_true: np.array, y_test: np.array) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_test),
        "f1": f1_score(y_true, y_test)
    }


def main():
    print("Main içindeyim.")
    breast_cancer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "breast_cancer.csv")
    df = pd.read_csv(breast_cancer_path)
    print("veriyi okudum.")
    # data = load_breast_cancer()
    # df = pd.DataFrame(data.data, columns=data.feature_names)
    # df["target"] = data.target
    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # mlflow.set_experiment("breast_cancer_docker_example_2")
    # client = mlflow.tracking.MlflowClient()
    # experiment_id = client.get_experiment_by_name("breast_cancer_docker_example_2").experiment_id
    # run = client.create_run(experiment_id)
    with mlflow.start_run():
        lr = LogisticRegression(solver="newton-cg")
        lr.fit(X_train, y_train)
        train_pred = lr.predict(X_train)
        test_pred = lr.predict(X_test)
        train_metrics = eval_metrics(y_train, train_pred)
        test_metrics = eval_metrics(y_test, test_pred)
        for key, value in train_metrics.items():
            mlflow.log_metric(f"train_{key}", value)
        for key, value in test_metrics.items():
            mlflow.log_metric(f"test_{key}", value)
        mlflow.sklearn.log_model(lr, "model")
        for key, value in lr.get_params().items():
            mlflow.log_param(key, value)
        mlflow.set_tags({"model_class": lr.__class__})
        mlflow.set_tags({"model_name": lr.__class__.__name__})
        mlflow.set_tags({"mlflow.user": "fatih"})
        mlflow.set_tags({"mlflow.runName": "LogisticRegression Model Run"})


if __name__ == "__main__":
    main()
