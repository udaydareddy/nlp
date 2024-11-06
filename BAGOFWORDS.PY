#import libraries 

import numpy as np
import pandas as pd

#read the dataset using pandas

dataset = pd.read_csv('email_classification.csv')
dataset.head()

#count th spam and ham mails from the dataset using the pandas library 

dataset.label.value_counts()

#create the separte column for the spam and ham emails i.e., 1 for 'spam' and 0 for 'ham' using lambda or we can use separate function 

dataset['spam'] = dataset['label'].apply(lambda x : 1 if x == 'spam' else 0)

#split the dataset into training and test sets

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset.email, dataset.spam, test_size=0.20)

#import the count vectorizer from the sklearn.feature_extraction.text module

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

#transfrom the X_trian and X_test data using the countvectorizer

X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)

#import the navies bayes claasifier the classification of spam and ham
 
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train_cv, y_train)#fit the model

#predict the test set results 

y_pred = classifier.predict(X_test_cv)

#compare the predict results with the test set results to know the model performance 
#use classification report and accuracy score to evaluate the performance

from sklearn.metrics import classification_report, accuracy_score
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
