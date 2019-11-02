import sys
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB

import numpy as np


X = np.array([ ['a'], ['a'], ['n'], ['n'], ['s'], [2]] )
y = np.array([ 1, 1, 0, 1, 1, 1 ]) 

lb = preprocessing.LabelBinarizer()
lb.fit(X)
X2 = lb.transform(X)

clf = MultinomialNB()
clf.fit(X2, y)

letter = sys.argv[1]

pred_me = lb.transform([ letter ] )
print(clf.predict_proba(  pred_me  ) )
print(clf.predict(pred_me))
