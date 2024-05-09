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

pd.set_option("display.max_columns", None)



class Feature_Construction(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()
        return

    def fit(self, X, y=None):
        return self

    def __feature_construction(self, X):
        X["BMI"] = X["weight"] / X["height"]
        X["do_physical_activity"] = X["physical_activity"] > 0
        X["up_age_25"] = X["age"] > 25
        X["over_all_meals"] = X["main_meals"] + X["frequency_vegetables"]

        def get(x):
            x = str(x)
            x = x[2:4]
            x = int(x)
            return x

        X["perfect_weight"] = abs(X["height"].apply(get) - X["weight"]) <= 5
        for col in X.columns:
            if X[col].dtype == "bool":
                X[col] = X[col].astype("float")
        return X

    def transform(self, X, y=None):
        X = self.__feature_construction(X)
        return X
    
class Encode(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()
        return

    def fit(self, X, y=None):
        return self

    def __dummy(self, X):
        features = [
            "gender",
            "family_history",
            "frequency_high_caloric_food",
            "eating_out_main_meals",
            "smoking",
            "calories_monitoring",
            "alcohol",
            "transportation",
        ]

        X = pd.get_dummies(data=X, columns=features, drop_first=True)
        return X

    def transform(self, X, y=None):
        X = self.__dummy(X)

        for col in X.columns:
            if X[col].dtype == "bool":
                X[col] = X[col].astype("float")
        return X


feature_construction = Feature_Construction()
encode = Encode()