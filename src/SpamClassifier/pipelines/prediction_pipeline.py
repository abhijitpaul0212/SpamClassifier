import os
import sys
import pandas as pd
from src.SpamClassifier.exception import CustomException
from src.SpamClassifier.logger import logging
from src.SpamClassifier.utils.utils import Utils


class PredictPipeline:
    def __init__(self):
        self.utils = Utils()
    
    def predict(self, features):
        try:
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model_path = os.path.join("artifacts", "model.pkl.gz")
            
            preprocessor = self.utils.load_object(preprocessor_path)
            model = self.utils.load_object(model_path)
            
            scaled_data = preprocessor.transform(features)            
            pred = model.predict(scaled_data)
            
            return pred
            
        except Exception as e:
            raise CustomException(e, sys)
    
    
class CustomData:
    def __init__(self, messages: str):
        self.messages = messages         
                
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'messages': [self.messages]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e, sys)