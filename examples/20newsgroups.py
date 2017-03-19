from __future__ import unicode_literals, division

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.datasets import fetch_20newsgroups_vectorized
from active_learning.active_learning import ActiveLearner
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy.sparse import vstack
import matplotlib.pyplot as plt
from copy import deepcopy
import seaborn as sns
import numpy as np


results_holder = {
    'least_confident': [],
    'max_margin': [],
    'entropy': [],
}

if __name__ == '__main__':
    all_uncertainty_sampling_results = deepcopy(results_holder)
    all_random_sampling_results = []
    data = fetch_20newsgroups_vectorized()
    clf = LogisticRegression()

    for _ in range(10):
        uncertainty_sampling_results = deepcopy(results_holder)
        num_samples, random_sampling_results = [], []

        X_train, X_test, y_train, y_test = train_test_split(
            data.data, data.target, test_size=0.5)
        
        X_labeled, X_unlabeled, y_labeled, y_oracle = train_test_split(
            X_train, y_train, test_size=0.8)

        for num_queries in (0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500):
            num_samples.append(num_queries)

            random_queries = np.random.choice(X_unlabeled.shape[0], num_queries, replace=False)
            X_augmented = vstack((X_labeled, X_unlabeled[random_queries, :]))
            y_augmented = np.concatenate((y_labeled, y_oracle[random_queries]))
            clf.fit(X_augmented, y_augmented)
            random_sampling_results\
                .append(np.sum(clf.predict(X_test) == y_test) / np.shape(X_test)[0])

            for strategy in uncertainty_sampling_results:
                clf.fit(X_labeled, y_labeled)
                idx = ActiveLearner(strategy=strategy).rank(clf, X_unlabeled, num_queries)
                X_augmented = vstack((X_labeled, X_unlabeled[idx, :]))
                y_augmented = np.concatenate((y_labeled, y_oracle[idx]))
                clf.fit(X_augmented, y_augmented)
                uncertainty_sampling_results[strategy]\
                    .append(np.sum(clf.predict(X_test) == y_test) / np.shape(X_test)[0])

        all_random_sampling_results.append(random_sampling_results)
        for strategy in uncertainty_sampling_results:
            all_uncertainty_sampling_results[strategy]\
                .append(uncertainty_sampling_results[strategy])


    sns.set_style("darkgrid")
    plt.plot(num_samples, np.mean(all_random_sampling_results, axis=0), 'red', 
             num_samples, np.mean(all_uncertainty_sampling_results['least_confident'], axis=0), 'blue',
             num_samples, np.mean(all_uncertainty_sampling_results['max_margin'], axis=0), 'green',
             num_samples, np.mean(all_uncertainty_sampling_results['entropy'], axis=0), 'orange',
    )
    plt.legend(['Random Sampling', 'Least Confident', 'Max Margin', 'Entropy'], loc=4)
    plt.ylabel('Accuracy'); plt.xlabel('Number of Queries'); plt.title('20 Newsgroups - Uncertainty Sampling'); plt.ylim([0,1])
    plt.savefig('misc/20newsgroups.jpg')
