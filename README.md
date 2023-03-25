# Naive Bayes Spam Filter: Project Overview
- Implemented a spam filter in Python using Naive Bayes from scratch (without the help of pandas and scikit-learn).

## Dataset

[SMS Spam Collection Data Set - UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)

## Data Preprocessing
- removed punctuation
- set all the letters to lower case
- split the senteces to distinct words
- randomised the dataset
- split the data into training data and test data
- separated spam SMS from ham SMS
- extracted the SMS from the data (removed the label)
- extracted all the words that appear in the SMS
- removed duplicates from the vocabulary

## Model Building
Below are the steps I took to build the model:
- Calculated the probability of finding a word in the spam/ham SMS
- Multiplied all the word probabilities together
- Used Naive Bayes to classify a SMS as spam

## Model Performance
Number of test data: 1672  
True Positive: 208  
True Negative: 1390  
False Positive: 39  
False Negative: 35  

Accuracy = 0.96  
Precision = 0.84  
Recall = 0.86  
F1 = 0.85  
