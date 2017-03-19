from __future__ import unicode_literals, division

from scipy.sparse import csc_matrix, vstack
from scipy.stats import entropy
from collections import Counter
import numpy as np


class ActiveLearner(object):
    """Determine the optimal querying strategy for unlabeled data. 

    Suppose you're given a small set of labeled points, a large set of
    unlabeled points, and in addition, you can request labels for n of your
    unlabeled points. Active Learning provides a framework for choosing the
    points whose labels will give us the most information.

    This class implements three types of uncertainty sampling: Least
    confident (query the instances about which your model is least confident),
    max margin (query the instances which have the smallest ratio between the
    model's top two predictions), and entropy (query the instances whose model
    output distributions have the most entropy).

    It also implements two types of query by committee: vote entropy (query
    instances where the entropy amongst votes is maximized) and average kl
    divergence (query instances of max kl divergence from the consensus).

    Parameters
    ----------
    num_queries : int or float or None, default=None
        Number of queries to rank. None for rank all points, float for rank
        percentage of unlabeled point, int for rank n unlabeled points.

    strategy : 'entropy', 'least_confident', 'max_margin', 'vote_entropy',
        'average_kl_divergence', default='least_confident'
        Strategy for ranking unlabeled points as canditates for querying.
    """

    _uncertainty_sampling_frameworks = [
        'entropy',
        'max_margin',
        'least_confident',
    ]

    _query_by_committee_frameworks = [
        'vote_entropy',
        'average_kl_divergence',
    ]

    def __init__(self, strategy='least_confident'):
        self.strategy = strategy

    def rank(self, clf, X_unlabeled, num_queries=None):
        """Rank unlabeled instances as querying candidates.

        Parameters
        ----------
        clf : classifier
            Pre-trained probabilistic classifier conforming to the sklearn
            interface.

        X_unlabeled : sparse matrix, [n_samples, n_features]
            Unlabeled training instances.

        Returns
        -------
        rankings : ndarray, shape (num_queries,)
            cluster labels
        """
        if num_queries == None:
            num_queries = X_unlabeled.shape[0]

        elif type(num_queries) == float:
            num_queries = int(num_queries * X_unlabeled.shape[0])

        if self.strategy in self._uncertainty_sampling_frameworks:
            scores = self.__uncertainty_sampling(clf, X_unlabeled)

        elif self.strategy in self._query_by_committee_frameworks:
            scores = self.__query_by_committee(clf, X_unlabeled)

        else: 
            raise ValueError(
                "I haven't implemented this strategy. Sorry."
            )

        rankings = np.argsort(-scores)[:num_queries]
        return rankings

    def __uncertainty_sampling(self, clf, X_unlabeled):
        probs = clf.predict_proba(X_unlabeled)

        if self.strategy == 'least_confident':
            return 1 - np.amax(probs, axis=1)

        elif self.strategy == 'max_margin':
            margin = np.partition(-probs, 1, axis=1)
            return -np.abs(margin[:,0] - margin[:, 1])

        elif self.strategy == 'entropy':
            return np.apply_along_axis(entropy, 1, probs)

    def __query_by_committee(self, clf, X_unlabeled):
        num_classes = len(clf[0].classes_)
        C = len(clf)
        preds = []

        if self.strategy == 'vote_entropy':
            for model in clf:
                y_out = map(int, model.predict(X_unlabeled))
                preds.append(np.eye(num_classes)[y_out])

            votes = np.apply_along_axis(np.sum, 0, np.stack(preds)) / C
            return np.apply_along_axis(entropy, 1, votes)

        elif self.strategy == 'average_kl_divergence':
            for model in clf:
                preds.append(model.predict_proba(X_unlabeled))

            consensus = np.mean(np.stack(preds), axis=0)
            divergence = []
            for y_out in preds:
                divergence.append(entropy(consensus.T, y_out.T))
            
            return np.apply_along_axis(np.mean, 0, np.stack(divergence))
