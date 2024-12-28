import pandas as pd
from sklearn.metrics import precision_recall_fscore_support as score
import mlflow
import warnings
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")
#models with grid search 
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support as score, accuracy_score
from sklearn.model_selection import GridSearchCV
import mlflow
import warnings
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

def RandomForestModelG(data_url, version, df, X_resampled, y_resampled, X_test, y_test):
    mlflow.sklearn.autolog(disable=True)
    with mlflow.start_run(run_name='RandomForest_Grid'):
        mlflow.log_param("data_url", data_url)
        mlflow.log_param("data_version", version)
        mlflow.log_param("input_rows", df.shape[0])
        mlflow.log_param("input_cols", df.shape[1])

        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5]
        }
        rf = GridSearchCV(RandomForestClassifier(random_state=5), param_grid, cv=3, scoring='accuracy')
        rf.fit(X_resampled, y_resampled)

        mlflow.set_tag(key="model", value="RandomForest_Grid")
        params = rf.best_params_
        mlflow.log_params(params)
        
        predicted = rf.predict(X_test)
        precision, recall, fscore, support = score(y_test, predicted, average='macro')
        accuracy = accuracy_score(y_test, predicted)
        mlflow.log_metric("Precision_test", precision)
        mlflow.log_metric("Recall_test", recall)
        mlflow.log_metric("F1_score_test", fscore)
        mlflow.log_metric("Accuracy_test", accuracy)
        mlflow.sklearn.log_model(rf.best_estimator_, artifact_path="ML_models")

def XGBoostModelG(data_url, version, df, X_train, y_train, X_test, y_test):
    mlflow.xgboost.autolog(disable=True)
    with mlflow.start_run(run_name='XGBoost_Grid'):
        mlflow.log_param("data_url", data_url)
        mlflow.log_param("data_version", version)
        mlflow.log_param("input_rows", df.shape[0])
        mlflow.log_param("input_cols", df.shape[1])

        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
            'learning_rate': [0.01, 0.1]
        }
        xg = GridSearchCV(XGBClassifier(), param_grid, cv=3, scoring='accuracy')
        xg.fit(X_train, y_train)

        mlflow.set_tag(key="model", value="XGBClassifier")
        params = xg.best_params_
        mlflow.log_params(params)

        predicted = xg.predict(X_test)
        precision, recall, fscore, support = score(y_test, predicted, average='macro')
        accuracy = accuracy_score(y_test, predicted)
        mlflow.log_metric("Precision_test", precision)
        mlflow.log_metric("Recall_test", recall)
        mlflow.log_metric("F1_score_test", fscore)
        mlflow.log_metric("Accuracy_test", accuracy)
        mlflow.xgboost.log_model(xg.best_estimator_, artifact_path="ML_models")

def logisticRegressionModelG(data_url, version, df, X_resampled, y_resampled, X_test, y_test):
    mlflow.sklearn.autolog(disable=True)
    with mlflow.start_run(run_name='LogisticRegression-Grid'):
        mlflow.log_param("data_url", data_url)
        mlflow.log_param("data_version", version)
        mlflow.log_param("input_rows", df.shape[0])
        mlflow.log_param("input_cols", df.shape[1])

        param_grid = {
            'C': [0.1, 1, 10],
            'solver': ['lbfgs', 'liblinear']
        }
        lr = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=3, scoring='accuracy')
        lr.fit(X_resampled, y_resampled)

        mlflow.set_tag(key="model", value="LogisticRegression")
        params = lr.best_params_
        mlflow.log_params(params)

        predicted = lr.predict(X_test)
        precision, recall, fscore, support = score(y_test, predicted, average='macro')
        accuracy = accuracy_score(y_test, predicted)
        mlflow.log_metric("Precision_test", precision)
        mlflow.log_metric("Recall_test", recall)
        mlflow.log_metric("F1_score_test", fscore)
        mlflow.log_metric("Accuracy_test", accuracy)
        mlflow.sklearn.log_model(lr.best_estimator_, artifact_path="ML_models")
#Svm model
def SVMModelG(data_url, version, df, X_resampled, y_resampled, X_test, y_test):
    mlflow.sklearn.autolog(disable=True)
    with mlflow.start_run(run_name='SVM_Grid'):
        mlflow.log_param("data_url", data_url)
        mlflow.log_param("data_version", version)
        mlflow.log_param("input_rows", df.shape[0])
        mlflow.log_param("input_cols", df.shape[1])

        param_grid = {
            'C': [0.1, 1, 10],
            'degree': [2, 3, 4],
            'gamma': ['scale', 'auto']
        }
        svm = GridSearchCV(SVC(kernel='poly'), param_grid, cv=3, scoring='accuracy')
        svm.fit(X_resampled, y_resampled)

        mlflow.set_tag(key="model", value="SVM")
        params = svm.best_params_
        mlflow.log_params(params)

        predicted = svm.predict(X_test)
        precision, recall, fscore, support = score(y_test, predicted, average='macro')
        accuracy = accuracy_score(y_test, predicted)
        mlflow.log_metric("Precision_test", precision)
        mlflow.log_metric("Recall_test", recall)
        mlflow.log_metric("F1_score_test", fscore)
        mlflow.log_metric("Accuracy_test", accuracy)
        mlflow.sklearn.log_model(svm.best_estimator_, artifact_path="ML_models")












#models without grid search and accuracy metric upload :
def RandomForestModel(data_url, version, df, X_resampled, y_resampled, X_test, y_test):
    mlflow.sklearn.autolog(disable=True)
    with mlflow.start_run(run_name='RandomForest'):
        mlflow.log_param("data_url", data_url)
        mlflow.log_param("data_version", version)
        mlflow.log_param("input_rows", df.shape[0])
        mlflow.log_param("input_cols", df.shape[1])
        rf = RandomForestClassifier(random_state=5)
        mlflow.set_tag(key="model", value="RandomForest")
        params = rf.get_params()
        mlflow.log_params(params)
        rf.fit(X_resampled, y_resampled)
        train_features_name = 'X_resampled'
        train_label_name = 'y_resampled'
        mlflow.set_tag(key="train_features_name", value=train_features_name)
        mlflow.set_tag(key="train_label_name", value=train_label_name)
        predicted = rf.predict(X_test)
        precision, recall, fscore, support = score(y_test, predicted, average='macro')
        mlflow.log_metric("Precision_test", precision)
        mlflow.log_metric("Recall_test", recall)
        mlflow.log_metric("F1_score_test", fscore)
        mlflow.sklearn.log_model(rf, artifact_path="ML_models")

def XGBoostModel(data_url, version, df, X_train, y_train, X_test, y_test):
    mlflow.xgboost.autolog(disable=True)
    with mlflow.start_run(run_name='XGBoost'):
        mlflow.log_param("data_url", data_url)
        mlflow.log_param("data_version", version)
        mlflow.log_param("input_rows", df.shape[0])
        mlflow.log_param("input_cols", df.shape[1])
        xg = XGBClassifier()
        params = xg.get_params()
        mlflow.set_tag(key="model", value="XGBClassifier")
        mlflow.log_params(params)
        xg.fit(X_train, y_train)
        train_features_name = 'X_train'
        train_label_name = 'y_train'
        mlflow.set_tag(key="train_features_name", value=train_features_name)
        mlflow.set_tag(key="train_label_name", value=train_label_name)
        predicted = xg.predict(X_test)
        precision, recall, fscore, support = score(y_test, predicted, average='macro')
        mlflow.log_metric("Precision_test", precision)
        mlflow.log_metric("Recall_test", recall)
        mlflow.log_metric("F1_score_test", fscore)
        mlflow.xgboost.log_model(xg, artifact_path="ML_models")

def logisticRegressionModel(data_url, version, df, X_resampled, y_resampled, X_test, y_test):
    mlflow.sklearn.autolog(disable=True)
    with mlflow.start_run(run_name='LogisticRegression'):
        mlflow.log_param("data_url", data_url)
        mlflow.log_param("data_version", version)
        mlflow.log_param("input_rows", df.shape[0])
        mlflow.log_param("input_cols", df.shape[1])
        lr = LogisticRegression(max_iter=1000, solver='lbfgs')
        mlflow.set_tag(key="model", value="LogisticRegression")
        params = lr.get_params()
        mlflow.log_params(params)
        lr.fit(X_resampled, y_resampled)
        train_features_name = 'X_resampled'
        train_label_name = 'y_resampled'
        mlflow.set_tag(key="train_features_name", value=train_features_name)
        mlflow.set_tag(key="train_label_name", value=train_label_name)
        predicted = lr.predict(X_test)
        precision, recall, fscore, support = score(y_test, predicted, average='macro')
        mlflow.log_metric("Precision_test", precision)
        mlflow.log_metric("Recall_test", recall)
        mlflow.log_metric("F1_score_test", fscore)
        mlflow.sklearn.log_model(lr, artifact_path="ML_models")