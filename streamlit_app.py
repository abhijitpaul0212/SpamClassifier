# streamlit_app.py

import streamlit as st
import pandas as pd

from src.SpamClassifier.pipelines.prediction_pipeline import PredictPipeline
from src.SpamClassifier.utils.utils import Utils
from src.SpamClassifier.utils.data_processor import CSVProcessor


# Design Streamlit Page
st.write("""
# Spam Classifier
This app classifies the **Spam Message**!
""")
st.write('---')

# Header of Specify Input Parameters
st.header('Specify Input Parameters')

def user_input_features():
    text_message = st.text_area("TEXT MESSAGE:")

    data = {
                'messages': text_message
            }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Main Panel
st.header('Specified Input parameters')
st.write(df)
st.write('---')

predict_pipeline = PredictPipeline()
prediction = predict_pipeline.predict(df)

st.header('Message is')
st.write("SPAM" if prediction[0] == 1 else "NOT A SPAM")

st.write('---')
