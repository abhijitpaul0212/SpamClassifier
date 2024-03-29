# model_evaluation.py

import os
import sys

import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from urllib.parse import urlparse
from dataclasses import dataclass

from src.SpamClassifier.logger import logging
from src.SpamClassifier.exception import CustomException
from src.SpamClassifier.utils.utils import Utils
from src.SpamClassifier.utils.mlflow_setup import setup_mlflow_experiment
import src.SpamClassifier.utils.mlflow_setup as mlflow_setup


@dataclass
class ModelEvaluation:

    def eval_metrics(self, model, features, label):
        accuracy = round(cross_val_score(model, features, label, cv=10, n_jobs=-1, scoring='accuracy').mean(), 2)
        f1 = round(cross_val_score(model, features, label, cv=10, n_jobs=-1, scoring='f1').mean(), 2)
        precision = round(cross_val_score(model, features, label, cv=10, n_jobs=-1, scoring='precision').mean(), 2)
        recall = round(cross_val_score(model, features, label, cv=10, n_jobs=-1, scoring='recall').mean(), 2)
        roc_auc = round(cross_val_score(model, features, label, cv=10, n_jobs=-1, scoring='roc_auc').mean(), 2)
        return accuracy, f1, precision, recall, roc_auc
    
    def initiate_model_evaluation(self, test_array):
        try:
            X_test, y_test = (test_array[:, :-1], test_array[:, -1])
            model_path = os.path.join("artifacts", "model.pkl.gz")
            model = Utils().load_object(model_path)
            model_name = str(model).split("(")[0]

            with mlflow.start_run(run_id=mlflow_setup.get_active_run_id(), nested=True):
                
                mlflow.set_tag("Best Model", model_name)
                
                accuracy, f1, precision, recall, roc_auc = self.eval_metrics(model, X_test, y_test)

                logging.info("Model: {}".format(model_name))
                logging.info("accuracy: {}".format(accuracy))
                logging.info("f1: {}".format(f1))
                logging.info("precision: {}".format(precision))
                logging.info("recall: {}".format(recall))
                logging.info("roc_auc: {}".format(roc_auc))

                mlflow.log_metric("accuracy score", accuracy)
                mlflow.log_metric("f1 score", f1)
                mlflow.log_metric("precision score", precision)
                mlflow.log_metric("recall score", recall)
                mlflow.log_metric("roc_auc score", roc_auc)
                mlflow.end_run()

                mlflow.sklearn.log_model(model, "model")
                
        except Exception as e:
            raise CustomException(e, sys)
