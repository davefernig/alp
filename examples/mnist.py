from __future__ import unicode_literals, division

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from active_learning.active_learning import ActiveLearner
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
from collections import Counter
from copy import deepcopy
import seaborn as sns
import numpy as np


results_holder = {
    'average_kl_divergence': [],
    'vote_entropy': [],
}

if __name__ == '__main__':
    all_query_by_commitee_results = deepcopy(results_holder)
    all_random_sampling_results = []
    mnist = fetch_mldata('MNIST original')    
    clf=[LogisticRegression() for _ in range(10)]
    
    for _ in range(10):
        query_by_commitee_results = deepcopy(results_holder)
        num_samples, random_sampling_results = [], []

        X_train, X_test, y_train, y_test = train_test_split(
            mnist.data, mnist.target, test_size=0.5)
        
        X_labeled, X_unlabeled, y_labeled, y_oracle = train_test_split(
            X_train, y_train, test_size=0.999)

        for num_queries in (10, 20, 30, 40, 50, 60, 80, 90, 100):
            num_samples.append(num_queries)
            random_queries = np.random.choice(X_unlabeled.shape[0], num_queries, replace=False)
            X_augmented = np.concatenate((X_labeled, X_unlabeled[random_queries, :]))
            y_augmented = np.concatenate((y_labeled, y_oracle[random_queries]))
            preds = []
            for model in clf:
                model.fit(X_augmented, y_augmented)
                preds.append(model.predict(X_test))
        
            majority_votes = np.apply_along_axis(lambda x: Counter(x).most_common()[0][0], 0, np.stack(preds))
            random_sampling_results.append(np.sum(majority_votes == y_test) / np.shape(X_test)[0])

            for strategy in query_by_commitee_results:
                for model in clf:
                    model.fit(X_labeled, y_labeled)

                AL = ActiveLearner(strategy=strategy)
                for model in clf:
                    model.classes_ = np.arange(10)
                idx = AL.rank(clf, X_unlabeled, num_queries)
                X_augmented = np.concatenate((X_labeled, X_unlabeled[idx, :]))
                y_augmented = np.concatenate((y_labeled, y_oracle[idx]))

                preds = []
                for model in clf:
                    model.fit(X_augmented, y_augmented)
                    preds.append(model.predict(X_test))
        
                majority_votes = np.apply_along_axis(lambda x: Counter(x).most_common()[0][0], 0, np.stack(preds))
                query_by_commitee_results[strategy].append(np.sum(majority_votes == y_test) / np.shape(X_test)[0])

        all_random_sampling_results.append(random_sampling_results)
        for strategy in query_by_commitee_results:
            all_query_by_commitee_results[strategy]\
                .append(query_by_commitee_results[strategy])
        
    sns.set_style("darkgrid")
    plt.plot(num_samples, np.mean(all_random_sampling_results, axis=0), 'red', 
             num_samples, np.mean(all_query_by_commitee_results['average_kl_divergence'], axis=0), 'blue',
             num_samples, np.mean(all_query_by_commitee_results['vote_entropy'], axis=0), 'green',
    )
    plt.legend(['Random Sampling', 'Average KL Divergence', 'Vote Entropy'], loc=4)
    plt.ylabel('Accuracy'); plt.xlabel('Number of Queries'); plt.title('MNIST - Query by commitee'); plt.ylim([0,1])
    plt.savefig('misc/mnist.jpg')
