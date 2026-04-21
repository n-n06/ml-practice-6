import os

import mlflow
import mlflow.xgboost as xgb
from mlflow.tracking import MlflowClient

def setup_mlflow():
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("bank-deposit-experiment")


def mlflow_run(train_func):
    def wrapper(*args, **kwargs):

        with mlflow.start_run():

            train_result = train_func(*args, **kwargs)

            params = train_result["params"]
            metrics = train_result["metrics"]
            best_model = train_result["model"]

            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

            xgb.log_model(
                best_model,
                artifact_path="model",
                registered_model_name="bank-deposit-model"
            )

            client = MlflowClient()

            latest_versions = client.get_latest_versions(
                "bank-deposit-model",
                stages=["None"]
            )

            if latest_versions:
                latest_version = latest_versions[0].version

                client.transition_model_version_stage(
                    name="bank-deposit-model",
                    version=latest_version,
                    stage="Production"
                )

        return train_result

    return wrapper
