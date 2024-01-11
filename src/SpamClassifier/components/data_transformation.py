# data_transformation.py

import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.SpamClassifier.logger import logging
from src.SpamClassifier.exception import CustomException
from src.SpamClassifier.utils.utils import Utils
from src.SpamClassifier.utils.data_processor import CSVProcessor
from src.SpamClassifier.utils.transformer import DataCleanser

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer

import warnings
warnings.filterwarnings("ignore")


@dataclass
class DataTransformationConfig:
    """
    This is configuration class for Data Transformation
    """
    preprocessor_obj_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    """
    This class handles Data Transformation
    """
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.utils = Utils()
        self.csv_processor = CSVProcessor()

    def transform_data(self):
        try:
            features = ['messages']

            text_preprocessing = Pipeline(
                steps=[
                    ('cleanser', DataCleanser()),
                    ('Vect', CountVectorizer(max_features=2500))
                ]
            )

            preprocessor = ColumnTransformer([
                ('text_preprocessing_pipeline', text_preprocessing, features)
            ], remainder='passthrough')

            return preprocessor
        
        except Exception as e:
            logging.error("Exception occured in Data Transformation")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path, val_path=None):
            
        try:
            train_df = self.utils.run_data_pipeline(self.csv_processor, filepath=None, filename=train_path)
            test_df = self.utils.run_data_pipeline(self.csv_processor, filepath=None, filename=test_path)
            
            logging.info(f'Train Dataframe Head : \n{train_df.head(2).to_string()}')
            logging.info(f'Test Dataframe Head : \n{test_df.head(2).to_string()}')
            
            target_column_name = 'label'
            drop_columns = [target_column_name]

            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = pd.get_dummies(train_df[target_column_name], drop_first=True).astype(int)

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = pd.get_dummies(test_df[target_column_name], drop_first=True).astype(int)

            logging.info("Applying preprocessing object on training, validation and testing datasets")
            preprocessing_obj = self.transform_data()
            preprocessing_obj.fit(input_feature_train_df)
            
            input_feature_train_arr = preprocessing_obj.transform(input_feature_train_df).toarray()
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df).toarray()       

            logging.info("Combining features and target columns into arrays")
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
     
            self.utils.save_object(
                file_path=self.data_transformation_config.preprocessor_obj_path,
                obj=preprocessing_obj
            )
            
            logging.info("Preprocessing pickle file saved")
            
            return (
                train_arr,
                test_arr
            )

        except Exception as e:
            logging.error("Exception occured in Initiate Data Transformation")
            raise CustomException(e, sys)
