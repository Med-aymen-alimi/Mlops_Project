import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# RandomForest Model
def RandomForestModel(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(random_state=5)
    rf.fit(X_train, y_train)

    predicted = rf.predict(X_test)
    accuracy = accuracy_score(y_test, predicted)

    print("Random Forest Model")
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:\n", classification_report(y_test, predicted))
    print("-" * 50)

# XGBoost Model
def XGBoostModel(X_train, y_train, X_test, y_test):
    xg = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xg.fit(X_train, y_train)

    predicted = xg.predict(X_test)
    accuracy = accuracy_score(y_test, predicted)

    print("XGBoost Model")
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:\n", classification_report(y_test, predicted))
    print("-" * 50)

# Logistic Regression Model
def LogisticRegressionModel(X_train, y_train, X_test, y_test):
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)

    predicted = lr.predict(X_test)
    accuracy = accuracy_score(y_test, predicted)

    print("Logistic Regression Model")
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:\n", classification_report(y_test, predicted))
    print("-" * 50)

# SVM Model
def SVMModel(X_train, y_train, X_test, y_test):
    svm = SVC(kernel='poly')
    svm.fit(X_train, y_train)

    predicted = svm.predict(X_test)
    accuracy = accuracy_score(y_test, predicted)

    print("SVM Model")
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:\n", classification_report(y_test, predicted))
    print("-" * 50)


