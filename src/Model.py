import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.base import BaseEstimator, TransformerMixin

# machine learning models
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# preprcessing

from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder,
    OneHotEncoder,
    MaxAbsScaler,
    MinMaxScaler,
)

from sklearn.feature_selection import (
    chi2,
    VarianceThreshold,
    f_classif,
    SelectKBest,
    SelectPercentile,
    SequentialFeatureSelector,
)
from sklearn.pipeline import Pipeline

# evaluation metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Neural networks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.activations import relu, softmax, linear

# heyperparameter tuning

from keras_tuner import RandomSearch, Hyperband, BayesianOptimization

import Processing as pr
import pickle

fc = pr.Feature_Construction()
enc = pr.Encode()


def RF(X, y, fc, en):
    rf = RandomForestClassifier(
        n_estimators=250,
        ccp_alpha=0.00015,
        random_state=42,
    )
    pip_rf = Pipeline(
        [("Feature_Construction", fc), ("Encode", en), ("RandomForestClassifier", rf)]
    )

    pip_rf.fit(X, y)
    return pip_rf


def XGB(X, y, fc, en):
    xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=8)
    leb_enc = LabelEncoder()
    y = leb_enc.fit_transform(y)
    pip_xgb = Pipeline(
        [("Feature_Construction", fc), ("Encode", en), ("XGBClassifier", xgb)]
    )

    pip_xgb.fit(X, y)
    return pip_xgb


def LGBM(X, y, fc, en):
    lgbm = LGBMClassifier(
        objective="multiclass",
        boosting_type="gbdt",
        num_class=7,
        learning_rate=0.01,
        n_estimators=500,
        max_depth=11,
    )
    pip_lgbm = Pipeline(
        [("Feature_Construction", fc), ("Encode", en), ("LGBMClassifier", lgbm)]
    )

    pip_lgbm.fit(X, y)
    return pip_lgbm

def vot(X, y):
    rf = RF(X, y, fc, enc)
    xgb = XGB(X, y, fc, enc)
    lgbm = LGBM(X, y, fc, enc)
    
    vot_system = VotingClassifier(
    estimators=[
            ("random", rf),
            ("xgbc", xgb),
            ("pip_lgbm", lgbm),
        ],
    )
    vot_system.fit(X, y)
    return vot_system


def save_model(model):
    with open("voting_classifier.pkl", "wb") as f:
        pickle.dump(model, f)
        

data = pd.read_csv("/home/ahmed/Ai/Data science and Ml projects/Multi-Class-Prediction-of-Obesity-Risk/Data set/train.csv")

def rename_columns(data, names):
    data.rename(columns=names, inplace=True)
    return data

columns_names = {
    "Gender": 'gender',
    'Age':"age",
    'Height':"height",
    'Weight':"weight",
    'family_history_with_overweight':"family_history",
    'FAVC':"frequency_high_caloric_food",
    'FCVC':"frequency_vegetables",
    'NCP':"main_meals",
    'CAEC':"eating_out_main_meals",
    'SMOKE':"smoking",
    'CH2O':"water_daily",
    'SCC':"calories_monitoring",
    'FAF':"physical_activity",
    'TUE':"technology_use",
    'CALC':"alcohol",
    'MTRANS':"transportation",
    'NObeyesdad':"obesity"
}

train_data = rename_columns(data, columns_names)

X = data.drop("obesity", axis=1)
y = data["obesity"]

save_model(vot(X, y))
