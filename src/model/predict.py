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
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        
        # Switched from releases to aliases - more future proof
        self.model_uri = "models:/bank-deposit-model@production"

        self.pipeline = self._load_model()

    def _load_model(self):
        try:
            return mlflow.sklearn.load_model(self.model_uri)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model from {self.model_uri}. "
                f"Ensure training has completed. Original error: {e}"
            ) from e

    def _to_frame(self, record) -> pd.DataFrame:
        if hasattr(record, "model_dump"):
            data = record.model_dump()
        else:
            data = dict(record)
        
        data = record.model_dump(mode="python")
        return pd.DataFrame([data])

    def predict(self, record) -> int:
        return int(self.pipeline.predict(self._to_frame(record))[0])

    def predict_proba(self, record) -> float:
        return float(self.pipeline.predict_proba(self._to_frame(record))[0][1])


