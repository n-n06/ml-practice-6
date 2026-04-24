import os

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

def setup_mlflow():
    uri = os.environ["MLFLOW_TRACKING_URI"]
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("bank-deposit-experiment")


def mlflow_run(train_func):
    """

    """
    def wrapper(*args, **kwargs):

        with mlflow.start_run():

            train_result = train_func(*args, **kwargs)

            params = train_result["params"]
            metrics = train_result["metrics"]
            best_model = train_result["model"]
            
            # get the schema
            schema = infer_signature(
                model_input=train_result["X_train"],
                model_output=train_result["y_train"]
            )
            
            # log hyperparams and the metrics
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

            model_info = mlflow.sklearn.log_model(
                best_model,
                artifact_path="model",
                registered_model_name="bank-deposit-model",
                signature=schema
            )
            latest_version = model_info.registered_model_version

            client = MlflowClient()

            client.set_registered_model_alias(
                name="bank-deposit-model",
                version=latest_version,
                alias="production",
            )

        return train_result

    return wrapper
