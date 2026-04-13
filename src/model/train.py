import sys
import os

sys.path.insert(
    0, 
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
)

import pandas as pd
import joblib
import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
 
from src.model.preprocess import build_preprocessing_pipeline
from src.model.config import RANDOM_STATE, MODEL_PATH 
 
 
def load_data() -> pd.DataFrame:
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "janiobachmann/bank-marketing-dataset",
        "bank.csv",
    )
    df["deposit"] = LabelEncoder().fit_transform(df["deposit"])
    return df
 
 
def build_full_pipeline() -> Pipeline:
    xgb = XGBClassifier(
        subsample=0.6,
        reg_lambda=2,
        reg_alpha=0.5,
        n_estimators=200,
        max_depth=7,
        learning_rate=0.05,
        gamma=1.5,
        colsample_bytree=0.7,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    )
    return Pipeline([
        ("preprocessor", build_preprocessing_pipeline()),
        ("classifier",   xgb),
    ])
 
 
def tune(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    param_grid = {
        "classifier__n_estimators": [180, 200, 220],
        "classifier__max_depth": [6, 7],
        "classifier__learning_rate": [0.04, 0.05, 0.06],
    }
 
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="accuracy",
        cv=3,
        verbose=1,
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
 
    print("Best parameters:", grid_search.best_params_)
    return grid_search.best_estimator_
 
 
def evaluate(pipeline: Pipeline, X_train, y_train, X_test, y_test) -> None:
    y_pred = pipeline.predict(X_test)
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
 

def main():
    df = load_data()
 
    X = df.drop(columns=["deposit"])
    y = df["deposit"]
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    pipeline = build_full_pipeline()
    best_pipeline = tune(pipeline, X_train, y_train)

    evaluate(best_pipeline, X_train, y_train, X_test, y_test)

    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
 
 
if __name__ == "__main__":
    main()
