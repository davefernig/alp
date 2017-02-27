from __future__ import unicode_literals, division
from scipy.sparse import csc_matrix
from scipy.stats import entropy
import numpy as np


class ActiveLearner(object):
    """Determine the optimal querying strategy for unlabeled data. 

    Suppose you're given a small set of labeled points, a large set of
    unlabeled points, and in addition, you can request labels for n of your
    unlabeled points. Active Learning provides a framework for choosing the
    points whose labels will give us the most information.

    This class implements three types of uncertainty sampling:

    Least Confident: Query the instances about which your model is least
        confident.

    Max Margin: Query the instances which have the smallest ratio between
        the model's top two predictions

    Entropy: Query the instances whose model output distributions have
        the most entropy.

    Parameters
    ----------
    num_queries : int or float or None, default=None
        Number of queries to rank. None for rank all points, float for rank
        percentage of unlabeled point, int for rank n unlabeled points.

    strategy : 'entropy', 'least_confident', or 'max_margin', default='entropy'
        Strategy for ranking unlabeled points as canditates for querying.
    """

    def __init__(self, strategy='entropy', num_queries=None):
        self.num_queries = num_queries
        self.strategy = strategy

    def rank(self, clf, X_unlabeled):
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
        if self.num_queries == None:
            num_queries = X_unlabeled.shape[0]

        elif type(self.num_queries) == int:
            num_queries = self.num_queries

        else:
            num_queries = int(self.num_queries * X_unlabeled.shape[0])

        probs = clf.predict_proba(X_unlabeled)

        if self.strategy == 'least_confident':
            scores = 1 - np.amax(probs, axis=1)

        elif self.strategy == 'max_margin':
            margin = np.partition(probs, 1, axis=1)
            scores = margin[:,0] - margin[:, 1]

        elif self.strategy == 'entropy':
            scores = np.apply_along_axis(entropy, 1, probs)

        else: 
            raise ValueError(
                "I haven't implemented this strategy. Sorry."
            )

        rankings = np.argsort(-scores)[:num_queries]
        return rankings
    
