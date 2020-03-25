import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

#1 Read in the dataset into a data frame
df = pd.read_csv("corpus.csv",encoding = "ISO-8859-1",nrows=3000)


#2 Split the data into Train Test Split
X = df["body"]
y = df["title"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=42)

#3 Count Vectorizern and TFIDF
vectorizer = TfidfVectorizer()
X_train_v = vectorizer.fit_transform(X_train)

#4 Training the Classifier
nb_clf = MultinomialNB()
nb_clf.fit(X_train_v,y_train)

#5 Prediction and Evaluation
predictions = nb_clf.predict(X_test)
print(metrics.accuracy_score(y_test,predictions))
