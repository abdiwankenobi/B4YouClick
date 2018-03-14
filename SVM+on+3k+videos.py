
# coding: utf-8

# In[32]:

import numpy as np
from sklearn import preprocessing, cross_validation, svm
import pandas as pd 


# In[72]:

#Threshhold value 1%
df = pd.read_csv('dataset_y1.csv')
df.head()


# In[75]:

#set X as data except Class label 
X = np.array(df.drop(['Class'],1))
#set y as data with Class label
y = np.array(df['Class'])
#split training and testing data
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

#perform SVM classification with probability
clf = svm.SVC(probability=True)
clf.fit(X_train, y_train)

#calculate accuracy running trained classification on test data
accuracy = clf.score(X_test, y_test)
print(accuracy)


# In[76]:

#run algorithm on validation video and supply results without class
example_measures=np.array([["6.666666666666667", "0.0", "2.197802197802198", "30", "4", "133", "125", "26", "3.2"]])

#14.285714285714286, 0, 0.0, 7, 0, 108, 97, 18, 1.0309278350515463
example_measures = example_measures.reshape(len(example_measures), -1)

#predict class of validation data with probability
prediction = clf.predict_proba(example_measures)
print("Prediction:", prediction)


# In[70]:

#run algorithm on validation video and supply results without class
example_measures=np.array([[ "0.0", "0", "0.0", "0", "0", "3", "3","7", "0.0"]])


#14.285714285714286, 0, 0.0, 7, 0, 108, 97, 18, 1.0309278350515463
example_measures = example_measures.reshape(len(example_measures), -1)

#predict class of validation data with probability
prediction = clf.predict_proba(example_measures)
print("Prediction:", prediction)


# In[71]:

print ("{:.16f}".format(float("9.99999900e-01")))
print ("{:.16f}".format(float("1.00000010e-07")))


# # Kernel sigmoid test

# In[13]:

clf = svm.SVC(probability=True, kernel='sigmoid')
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test
print(accuracy)
example_measures=np.array([["6.666666666666667", "0.0", "2.197802197802198", "30", "4", "133", "125", "26", "3.2"]])
example_measures = example_measures.reshape(len(example_measures), -1)

prediction = clf.predict_proba(example_measures)
print("Prediction:", prediction)


# # Linear kernel test

# In[20]:

clf = svm.SVC(probability=True, kernel='linear')
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)
example_measures=np.array([["6.666666666666667", "0.0", "2.197802197802198", "30", "4", "133", "125", "26", "3.2"]])
example_measures = example_measures.reshape(len(example_measures), -1)

prediction = clf.predict_proba(example_measures)
print("Prediction:", prediction)


# In[18]:

print ("{:.16f}".format(float("4.88581583e-11")))
print ("{:.16f}".format(float("1.00000000e+00")))


# # Polynomial Kernel test

# In[21]:

clf = svm.SVC(probability=True, kernel='poly')
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)
example_measures=np.array([["6.666666666666667", "0.0", "2.197802197802198", "30", "4", "133", "125", "26", "3.2"]])
example_measures = example_measures.reshape(len(example_measures), -1)

prediction = clf.predict_proba(example_measures)
print("Prediction:", prediction)


# In[19]:

print ("{:.16f}".format(float("3.00000090e-14")))
print ("{:.16f}".format(float("1.00000000e+00")))


# # Threshold value testing

# In[4]:

#Threshhold value 3%
df = pd.read_csv('dataset_y3.csv')
df.head()


# In[7]:

X = np.array(df.drop(['Class'],1))
y = np.array(df['Class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

clf = svm.SVC()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)


# In[8]:

example_measures = np.array([["14.285714","0.0","0.0", "7","10","29","35","8","2.857143"], ["0.000000","30.0","0.0","9","10","37","50","9","6.000000"]])

example_measures = example_measures.reshape(len(example_measures), -1)

prediction = clf.predict(example_measures)
print("Prediction should be [0,1]:", prediction)


# In[16]:

#Threshhold value 5%
df = pd.read_csv('dataset_y5.csv')
df.head()


# In[19]:

X = np.array(df.drop(['Class'],1))
y = np.array(df['Class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

clf = svm.SVC()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)


# In[21]:

example_measures = np.array([["14.285714","0.0","0.0", "7","10","29","35","8","2.857143"], ["0.000000","30.0","0.0","9","10","37","50","9","6.000000"]])

example_measures = example_measures.reshape(len(example_measures), -1)

prediction = clf.predict(example_measures)
print("Prediction should be [0,1]:", prediction)


# In[22]:

#Threshhold value 1.5%
df = pd.read_csv('dataset_y1_5.csv')
df.head()


# In[27]:

X = np.array(df.drop(['Class'],1))
y = np.array(df['Class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

clf = svm.SVC()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)


# In[49]:

example_measures = np.array([["14.285714","0.0","0.0", "7","10","29","35","8","2.857143"], ["0.000000","30.0","0.0","9","10","37","50","9","6.000000"],
                            ["0","16.66666667","0","27","6","137","101","10","0.99009901"]])

example_measures = example_measures.reshape(len(example_measures), -1)

prediction = clf.predict(example_measures)
print("Prediction should be [1,1,0]:", prediction)


# In[30]:

#Threshhold value 0.5%
df = pd.read_csv('dataset_y0_5.csv')
df.head()


# In[43]:

X = np.array(df.drop(['Class'],1))
y = np.array(df['Class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

clf = svm.SVC()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)


# In[46]:

example_measures = np.array([["14.285714","0.0","0.0", "7","10","29","35","8","2.857143"], 
                             ["0.000000","30.0","0.0","9","10","37","50","9","6.000000"],
                            ["0","16.66666667","0","27","6","137","101","10","0.99009901"]])

example_measures = example_measures.reshape(len(example_measures), -1)

prediction = clf.predict(example_measures)
print("Prediction should be [1,1,0]:", prediction)


# # Cross validation test: Linear vs Polynomial

# In[23]:

#set X as data except Class label 
X = np.array(df.drop(['Class'],1))
#set y as data with Class label
y = np.array(df['Class'])
#split training and testing data
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

#perform SVM classification with probability
clf = svm.SVC(probability=True)
clf.fit(X_train, y_train)

#calculate accuracy running trained classification on test data
accuracy = clf.score(X_test, y_test)
print(accuracy)
scores = cross_validation.cross_val_score(clf, X_train, y_train, cv=5)
#print('score: %0.5f (+/- %0.5f)' %scores.mean(), scores.std())


# In[24]:

print(scores)


# In[25]:

#perform SVM classification with probability
clf = svm.SVC(probability=True, kernel='linear')
clf.fit(X_train, y_train)

#calculate accuracy running trained classification on test data
accuracy = clf.score(X_test, y_test)
print(accuracy)
#apply cross validation with 5
scores = cross_validation.cross_val_score(clf, X_train, y_train, cv=5)
print(scores)


# In[26]:

#perform SVM classification with probability
clf = svm.SVC(probability=True, kernel='sigmoid')
clf.fit(X_train, y_train)

#calculate accuracy running trained classification on test data
accuracy = clf.score(X_test, y_test)
print(accuracy)
scores = cross_validation.cross_val_score(clf, X_train, y_train, cv=5)
print(scores)


# In[27]:

#perform SVM classification with probability
clf = svm.SVC(probability=True, kernel='poly')
clf.fit(X_train, y_train)

#calculate accuracy running trained classification on test data
accuracy = clf.score(X_test, y_test)
print(accuracy)
#apply cross validation with 5
scores = cross_validation.cross_val_score(clf, X_train, y_train, cv=5)
print(scores)

