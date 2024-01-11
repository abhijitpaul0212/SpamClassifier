# model_trainer.py

import os
import sys
import numpy as np
from dataclasses import dataclass
from src.SpamClassifier.logger import logging
from src.SpamClassifier.exception import CustomException
from src.SpamClassifier.utils.utils import Utils
from sklearn.naive_bayes import GaussianNB, MultinomialNB
import warnings
warnings.filterwarnings("ignore")


@dataclass
class ModelTrainerConfig:
    """
    This is configuration class FOR Model Trainer
    """
    trained_model_obj_path: str = os.path.join("artifacts", "model.pkl.gz")
    trained_model_report_path: str = os.path.join('artifacts', 'model_report.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.utils = Utils()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Splitting Dependent and Independent features from train and validation & test dataset")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1])
            
            models = {
                    'GaussianNB': GaussianNB(),
                    'MultinomialNB': MultinomialNB()
                }
            
            models = {
                     'GaussianNB': (GaussianNB(), {
                                 'var_smoothing': np.logspace(0, -20, num=100)}),
                     'MultinomialNB': (MultinomialNB(), {
                                          'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]})
                    }

            # model evaluation without any hyper-paramter tuning            
            # best_model = self.utils.evaluate_models(models, X_train, y_train, X_test, y_test, metric="accuracy")
            
            # model evaluation along with hyper-paramter tuning
            best_model = self.utils.evaluate_models_with_hyperparameter(models, X_train, y_train, X_test, y_test, metric="roc_auc", verbose=3)
            
            self.utils.save_object(
                 file_path=self.model_trainer_config.trained_model_obj_path,
                 obj=best_model
            )       

        except Exception as e:
            raise CustomException(e, sys)
