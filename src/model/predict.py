"""
Loads the saved pipeline and run inference

Typical usage (as a library):
    from predict import Predictor
    predictor = Predictor()
    result = predictor.predict(record)          # single dict => 0 or 1
    proba  = predictor.predict_proba(record)    # single dict => float (P(deposit=1))

"""
import os
import subprocess
import json
import sys

import pandas as pd
import mlflow


class Predictor:
    def __init__(self):
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))

        self.model_uri = "models:/bank-deposit-model/Production"

        self.pipeline = self._load_or_train()

    def _load_or_train(self):
        try:
            return mlflow.xgboost.load_model(self.model_uri)

        except Exception as e:
            subprocess.run(
                ["uv", "run", "src/model/train.py"],
                check=True
            )

            return mlflow.xgboost.load_model(self.model_uri)

    def _to_frame(self, record) -> pd.DataFrame:
        if hasattr(record, "model_dump"):
            data = record.model_dump()
        else:
            data = dict(record)

        data = {k: (v.value if hasattr(v, "value") else v) for k, v in data.items()}
        return pd.DataFrame([data])

    def predict(self, record) -> int:
        return int(self.pipeline.predict(self._to_frame(record))[0])

    def predict_proba(self, record) -> float:
        return float(self.pipeline.predict_proba(self._to_frame(record))[0][1])


if __name__ == "__main__":
    raw = sys.stdin.read().strip()
    record = json.loads(raw)
    predictor = Predictor()
    print(json.dumps({
        "prediction": predictor.predict(record),
        "probability": predictor.predict_proba(record),
    }))
