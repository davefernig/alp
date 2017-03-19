from __future__ import unicode_literals, division

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_mldata
from collections import Counter
import numpy as np
import tempfile
import sys
import os

from sklearn.neural_network import MLPClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from active_learning.active_learning import ActiveLearner


mnist = fetch_mldata('MNIST original')
X_train, X_test, y_train, y_test = train_test_split(
    mnist.data, mnist.target, test_size=0.5, random_state=42)

X_labeled, X_unlabeled, y_labeled, y_oracle = train_test_split(
    X_train, y_train, test_size=0.999, random_state=42)
clf=[]
for i in range(10):
    mlp = LogisticRegression()
    clf.append(mlp)

AL = ActiveLearner(strategy='average_kl_divergence')

#for num_queries in (0.01, 0.02, 0.03, 0.04, 0.05):

print 'base of', X_labeled.shape[0], 'points'

for num_queries in (10, 20, 30, 40, 50, 60):
    random_sampling_results = []

    for _ in range(1):
        #random_queries = np.random.choice(X_unlabeled.shape[0],
        #    int(X_train.shape[0] * num_queries), replace=False)
        random_queries = np.random.choice(X_unlabeled.shape[0],
            num_queries, replace=False)

        X_augmented = np.concatenate((X_labeled, X_unlabeled[random_queries, :]))
        y_augmented = np.concatenate((y_labeled, y_oracle[random_queries]))
        print 'now training on', X_augmented.shape[0], 'points with', num_queries, 'random queries'
        preds = []
        for model in clf:
            model.fit(X_augmented, y_augmented)
            preds.append(model.predict(X_test))

        majority_votes = np.apply_along_axis(lambda x: Counter(x).most_common()[0][0], 0, np.stack(preds))
        random_sampling_results.append(np.sum(majority_votes == y_test) / np.shape(X_test)[0])

    print 'random:',  np.mean(random_sampling_results)

    for model in clf:
        model.fit(X_labeled, y_labeled)

    #idx = AL.rank(clf, X_unlabeled, int(X_train.shape[0] * num_queries))
    idx = AL.rank(clf, X_unlabeled, num_queries)
    X_augmented = np.concatenate((X_labeled, X_unlabeled[idx, :]))
    y_augmented = np.concatenate((y_labeled, y_oracle[idx]))
    print 'now training on', X_augmented.shape[0],  'points with', num_queries, 'active queries'
    preds = []
    for model in clf:
        model.fit(X_augmented, y_augmented)
        preds.append(model.predict(X_test))

    majority_votes = np.apply_along_axis(lambda x: Counter(x).most_common()[0][0], 0, np.stack(preds))
    print 'active learning:', np.sum(majority_votes == y_test) / np.shape(X_test)[0]
    
    
