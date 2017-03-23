# alp: active learning in python

This is a python implementation of some popular active learning techniques,
including uncertainty sampling and query-by-committee.
It is built on top of numpy, scipy, and sklearn.
I wrote this for my own learning purposes; it is not particularly efficient.

<p align="center">
  <img src="img/plot.jpg" alt=""/>
</p>


## Example
```python
from active_learning.active_learning import ActiveLearner
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

X, X_unlabeled, y, y_oracle = train_test_split(*make_classification())
clf = LogisticRegression().fit(X, y)

AL = ActiveLearner(strategy='entropy')
AL.rank(clf, X_unlabeled, num_queries=5)
```
I did this while working through Burr Settles' excellent
[literature survey](http://burrsettles.com/pub/settles.activelearning.pdf).
If you're interested in this topic you should read it.
