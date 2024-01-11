# transformer.py

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
        nltk.download("stopwords")

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
