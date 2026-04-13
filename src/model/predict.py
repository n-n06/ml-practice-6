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
import joblib

from src.model.config import MODEL_PATH

class Predictor:
    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            subprocess.run(["uv", "run",  "src/model/train.py"], check=True)

        self.pipeline = joblib.load(model_path)
 
    def _to_frame(self, record) -> pd.DataFrame:
        if hasattr(record, "model_dump"):
            data = record.model_dump()
        else:
            data = dict(record)
 
        data = {k: (v.value if hasattr(v, "value") else v) for k, v in data.items()}
        return pd.DataFrame([data])
 
    def predict(self, record) -> int:
        """Returns 0 (no deposit) or 1 (deposit)."""
        return int(self.pipeline.predict(self._to_frame(record))[0])
 
    def predict_proba(self, record) -> float:
        """Returns P(deposit=1)."""
        return float(self.pipeline.predict_proba(self._to_frame(record))[0][1])


if __name__ == "__main__":
    raw = sys.stdin.read().strip()
    record = json.loads(raw)
    predictor = Predictor()
    print(json.dumps({
        "prediction": predictor.predict(record),
        "probability": predictor.predict_proba(record),
    }))
