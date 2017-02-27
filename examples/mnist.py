from __future__ import unicode_literals, division

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.datasets import fetch_mldata
import numpy as np
import tempfile
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from active_learning.active_learning import ActiveLearner


mnist = fetch_mldata('MNIST original')
X_train, X_test, y_train, y_test = train_test_split(
    mnist.data, mnist.target, test_size=0.5, random_state=42)

X_labeled, X_unlabeled, y_labeled, y_oracle = train_test_split(
    X_train, y_train, test_size=0.9994, random_state=42)

clf = LogisticRegression()

AL = ActiveLearner(strategy='max_margin')

#for num_queries in (0.01, 0.02, 0.03, 0.04, 0.05):

print 'base of', X_labeled.shape[0], 'points'

for num_queries in (10, 20, 30, 40, 50, 60):
    random_sampling_results = []

    print 'now training with', num_queries, 'random queries'
    for _ in range(20):
        #random_queries = np.random.choice(X_unlabeled.shape[0],
        #    int(X_train.shape[0] * num_queries), replace=False)
        random_queries = np.random.choice(X_unlabeled.shape[0],
            num_queries, replace=False)

        X_augmented = np.concatenate((X_labeled, X_unlabeled[random_queries, :]))
        y_augmented = np.concatenate((y_labeled, y_oracle[random_queries]))
        clf.fit(X_augmented, y_augmented)
        random_sampling_results.append(np.sum(clf.predict(X_test) == y_test) / np.shape(X_test)[0])

    print 'random:',  np.mean(random_sampling_results)

    clf.fit(X_labeled, y_labeled)
    #idx = AL.rank(clf, X_unlabeled, int(X_train.shape[0] * num_queries))
    idx = AL.rank(clf, X_unlabeled, num_queries)
    X_augmented = np.concatenate((X_labeled, X_unlabeled[idx, :]))
    y_augmented = np.concatenate((y_labeled, y_oracle[idx]))
    print 'now training on', X_augmented.shape[0],  'points with', num_queries, 'active queries'
    clf.fit(X_augmented, y_augmented)
    print 'active learning:', np.sum(clf.predict(X_test) == y_test) / np.shape(X_test)[0]
    
    
