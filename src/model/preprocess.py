import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler, RobustScaler
)

# transform constants
BOOL_MAP = {"no": 0, "yes": 1}
MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
EDUCATION_MAP = {"unknown": 0, "primary": 1, "secondary": 2, "tertiary": 3}
CONTACT_MAP = {"unknown": 0, "telephone": 1, "cellular": 2}
POUTCOME_MAP = {"success": 3, "failure": 2, "unknown": 1, "other": 0}


DROP_COLS = ["poutcome", "pdays", "previous", "contact", "job_unknown",
             "duration", "day", "month"]


class BankPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.age_scaler = MinMaxScaler()
        self.balance_scaler = RobustScaler()
        self.duration_scaler = MinMaxScaler()

        self.train_columns_ = None

        self.edu_map = EDUCATION_MAP
        self.bool_map = BOOL_MAP
        self.contact_map = CONTACT_MAP
        self.month_map = MONTH_MAP 
        self.poutcome_map = POUTCOME_MAP

    def fit(self, X, y=None):
        X = X.copy()

        # scaling
        self.age_scaler.fit(X[['age']])
        self.balance_scaler.fit(X[['balance']])
        self.duration_scaler.fit(X[['duration']])

        # one-hot reference columns
        X_tmp = pd.get_dummies(X[['job', 'marital']], prefix=['job', 'marital'])
        self.train_columns_ = sorted(X_tmp.columns)

        return self

    def transform(self, X):
        X = X.copy()

        # sclaing
        X['age'] = self.age_scaler.transform(X[['age']])
        X['balance'] = self.balance_scaler.transform(X[['balance']])
        X['duration_scaled'] = self.duration_scaler.transform(X[['duration']])

        # custom label encoding
        X['education'] = X['education'].map(self.edu_map)
        X['default'] = X['default'].map(self.bool_map)
        X['housing'] = X['housing'].map(self.bool_map)
        X['loan'] = X['loan'].map(self.bool_map)
        X['contact_encoded'] = X['contact'].map(self.contact_map)

        # transform cyclic features
        X['month'] = X['month'].map(self.month_map)

        X['month_sin'] = np.sin(2 * np.pi * X['month'] / 12)
        X['month_cos'] = np.cos(2 * np.pi * X['month'] / 12)

        X['day_sin'] = np.sin(2 * np.pi * X['day'] / 31)
        X['day_cos'] = np.cos(2 * np.pi * X['day'] / 31)

        # map binary features
        X['was_contacted_before'] = (X['previous'] > 0).astype(int)

        # map
        X['poutcome_encoded'] = X['poutcome'].map(self.poutcome_map)
            
        # OHE
        dummies = pd.get_dummies(X[['job', 'marital']], prefix=['job', 'marital'])

        for col in self.train_columns_:
            if col not in dummies.columns:
                dummies[col] = 0

        dummies = dummies[self.train_columns_]

        X = X.drop(columns=['job', 'marital'])
        X = pd.concat([X, dummies], axis=1)
        
        # drop cols
        X = X.drop(columns=[c for c in DROP_COLS if c in X.columns])

        X = X.reindex(sorted(X.columns), axis=1)

        return X


def build_preprocessing_pipeline():
    return Pipeline([
        ("preprocessor", BankPreprocessor()),
    ])
