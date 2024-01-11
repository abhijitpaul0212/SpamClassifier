# Email-Spam-Classifier-Using-Naive-Bayes

![image](https://github.com/abhijitpaul0212/SpamClassifier/assets/9966441/041a0991-ea6c-439a-8e95-8f82764f08af)

### Naive Bayes
Naive Bayes is a supervised classification technique based on Bayes' Theorem with an assumption of independence among predictors. That is, a Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature.

It is a popular technique for text categorization, judging documents as belonging to one category or the other (such as spam or legitimate, sports or politics, etc.) with word frequencies as features.

**Goal**: Previously unseen records should be assigned a class as accurately as possible

* We have a bunch of emails classified as 'spam' and a bunch of emails classified as 'ham' (not spam)
* The emails are first read and stored in a dataframe. They are then parsed using CountVectorizer
* This information is used to train the model and its prediction is then tested with a sample input

**Python Libraries used**: pandas, numpy, io, os, CountVectorizer and MultinomialNB from sklearn

The Spam classifier classifies the given input as a spam/ham. 

### Prod Link
https://spam-ham-classifier.streamlit.app/

### DagsHub Experiment Tracking
![image](https://github.com/abhijitpaul0212/SpamClassifier/assets/9966441/264564b8-8318-49e3-bc22-e7ceeb4fd335)

### Some Practical Applications:
* Direct Marketing
* Fraud Detection
* Text Classification
* Spam Filtering
* Categorizing News
* Medical Diagnosis
* Face Recognition
