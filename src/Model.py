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

import pickle

preprocessing = pickle.load(open("models_processing_pipeline.pkl", "rb"))

