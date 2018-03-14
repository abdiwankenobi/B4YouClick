
# coding: utf-8

# In[1]:

import numpy as np
from sklearn import preprocessing, cross_validation, svm
import pandas as pd 


# In[2]:

df = pd.read_csv('youtubeDF1.csv')
df.head()


# In[22]:

#set X as data except Class label 
X = np.array(df.drop(['Class'],1))
#set y as data with Class label
y = np.array(df['Class'])
#split training and testing data
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

#perform SVM classification
clf = svm.SVC()
clf.fit(X_train, y_train)

#calculate accuracy running trained classification on test data
accuracy = clf.score(X_test, y_test)
print(accuracy)


# In[24]:

#run algorithm on validation video and supply results without class
example_measures=np.array([["6.666666666666667", "0.0", "2.197802197802198", "30", "4", "133", "125", "26", "3.2"]])
example_measures = example_measures.reshape(len(example_measures), -1)

#predict class of validation data
prediction = clf.predict(example_measures)
print("Prediction should be [1]:", prediction)


# # Threshhold value : 3%

# In[3]:

df = pd.read_csv('youtubeDF3.csv')

df.head()


# In[4]:

X = np.array(df.drop(['Class'],1))
y = np.array(df['Class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

clf = svm.SVC()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)


# In[47]:

example_measures = np.array([["14.285714","0.0","0.0", "7","10","29","35","8","2.857143"], ["0.000000","30.0","0.0","9","10","37","50","9","6.000000"]])

example_measures = example_measures.reshape(len(example_measures), -1)

prediction = clf.predict(example_measures)
print("Prediction should be [0,1]:", prediction)


# # Threshhold value 5%

# In[24]:

df = pd.read_csv('youtubeDF5.csv')
df.head()


# In[30]:

X = np.array(df.drop(['Class'],1))
y = np.array(df['Class'])
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)
clf = svm.SVC()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)


# In[34]:

example_measures = np.array([["14.285714","0.0","0.0", "7","10","29","35","8","2.857143"], ["0.000000","30.0","0.0","9","10","37","50","9","6.000000"]])
example_measures = example_measures.reshape(len(example_measures), -1)
prediction = clf.predict(example_measures)
print("Prediction should be [0,1]:", prediction)


# # Thresshold value : 7%

# In[48]:

df = pd.read_csv('youtubeDF7.csv')
df.head()


# In[60]:

X = np.array(df.drop(['Class'],1))
y = np.array(df['Class'])
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)
clf = svm.SVC()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)


# In[65]:

example_measures = np.array([["14.285714","0.0","0.0", "7","10","29","35","8","2.857143"], ["0.000000","30.0","0.0","9","10","37","50","9","6.000000"]])
example_measures = example_measures.reshape(len(example_measures), -1)
prediction = clf.predict(example_measures)
print("Prediction should be [0,0]:", prediction)


# # Thresshold Value: 9%

# In[66]:

df = pd.read_csv('youtubeDF9.csv')
df.head()


# In[80]:

X = np.array(df.drop(['Class'],1))
y = np.array(df['Class'])
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)
clf = svm.SVC()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)


# In[81]:

example_measures = np.array([["14.285714","0.0","0.0", "7","10","29","35","8","2.857143"], ["0.000000","30.0","0.0","9","10","37","50","9","6.000000"]])
example_measures = example_measures.reshape(len(example_measures), -1)
prediction = clf.predict(example_measures)
print("Prediction should be [0,0]:", prediction)

