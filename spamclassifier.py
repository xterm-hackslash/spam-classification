

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import string
import matplotlib.pyplot as plt
import numpy as np



dset = pd.read_csv("spam.csv",sep='\t',names=['Class','Message'])
dinfo=dset.info()
dinfo
dset.describe()
dset['Length'] = dset['Message'].apply(len)
dset.groupby('Class').count()
dset[dset['Length']==910]['Message'].iloc[0]
dObject = dset['Class'].values
dObject
dset.loc[dset['Class']=="ham","Class"] = 1

dset.loc[dset['Class']=="spam","Class"] = 0
dObject2=dset['Class'].values.astype(np.int64)

##Cleaning the punctuations
def cleanMessage(message):
    nonPunc = [char for char in message if char not in string.punctuation]
    nonPunc = "".join(nonPunc)
    return nonPunc

dset['Message'] = dset['Message'].apply(cleanMessage)
CV = CountVectorizer(stop_words="english")
xSet = dset['Message'].values
ySet = dset['Class'].values.astype(np.int64)

##Splitting of test and training data
xSet_train,xSet_test,ySet_train,ySet_test = train_test_split(xSet,ySet,test_size=0.2)
xSet_train_CV = CV.fit_transform(xSet_train)
NB = MultinomialNB()
NB.fit(xSet_train_CV,ySet_train)
xSet_test_CV = CV.transform(xSet_test)
ySet_predict = NB.predict(xSet_test_CV)

#ACCURACY
accuracyScore = accuracy_score(ySet_test,ySet_predict)*100
print("Prediction Accuracy :",accuracyScore)

####Application interface to check the model
msg = input("Enter Message: ")
msgInput = CV.transform([msg])
predict = NB.predict(msgInput)
if(predict[0]==0):
    print("------------------------MESSAGE-SENT-[CHECK-SPAM-FOLDER]---------------------------")
else:
    print("---------------------------MESSAGE-SENT-[CHECK-INBOX]------------------------------")