# transformer.py

import os
import numpy as np
import pandas as pd
from scipy import stats
from imblearn.over_sampling import SMOTE
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.base import BaseEstimator, TransformerMixin


class DataCleanser(BaseEstimator, TransformerMixin):
    def __init__(self):
        try:
            stopwords.words('english')
        except LookupError:
            # Specify the path where NLTK should download resources
            cwd = os.getcwd()

            # Specify the path within the root directory
            nltk_data_path = os.path.join(cwd, 'nltk_data')

            # Create the directory if it doesn't exist
            if not os.path.exists(nltk_data_path):
                os.makedirs(nltk_data_path)
                
            nltk.data.path.append(nltk_data_path)
            nltk.download('stopwords')

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Replace values outside the bounds with the bounds
        ps = PorterStemmer()
        corpus = []
        for i in range(0, len(X)):
            review = re.sub('[^a-zA-Z]', ' ', X['messages'][i])
            review = review.lower()
            review = review.split()

            review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
            review = " ".join(review)
            corpus.append(review)
        return corpus
