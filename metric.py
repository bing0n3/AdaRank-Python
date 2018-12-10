import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import make_scorer
from sklearn.utils import check_X_y
import sys
import math as math


def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    # return np.sum(r)

    gain = 2 ** r - 1
    discounts = np.log2(np.arange(len(r)) + 2)

    return np.sum(gain / discounts)
    
            


def ndcg_at_k(r, k, method=0):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max





class NDCGScorer:

    def __init__(self, K=5):
        self.K = K

    def __call__(self, y_true, y_predict):
        sorted_y_index = np.argsort(y_true)
        y = np.take(y_predict, sorted_y_index)
        return ndcg_at_k(y,self.K)

class NDCGScorer_qid:
    def __init__(self, K=5):
        self.scorer = NDCGScorer(K)
        self.K = K
    def __call__(self, y_true, y_predict, qid):
        return self.get_scores(y_true, y_predict, qid)

    def get_scores(self, true_list, pred_l, qid):
        scores = []
        # get group off-set
        prv_qid = qid[0]
        mark_index = [0]
        for itr, a in enumerate(qid):
            if a != prv_qid:
                mark_index.append(itr)
                prv_qid = a
        mark_index.append(qid.shape[0])
        for start, end in zip(mark_index, mark_index[1:]):
            # print(true_list[start:end])
            scores.append(self.scorer(true_list[start:end], pred_l[start:end]))

        return np.array(scores)
    
    


class map_scorer():
    
    def __call__(self, y_true, y_predict, qid):
        return self.get_scores(y_true, y_predict, qid)
    
    def get_scores(self, true_list, pred_l,qid):
        scores = []
        # get group off-set
        prv_qid = qid[0]
        mark_index = [0]
        for itr, a in enumerate(qid):
            if a != prv_qid:
                mark_index.append(itr)
                prv_qid = a
        mark_index.append(qid.shape[0])
        for start, end in zip(mark_index, mark_index[1:]):
            # print(true_list[start:end])
            scores.append(get_ap(true_list[start:end], pred_l[start:end]))
        return np.array(scores)


def get_ap(y_true, y_pred):
    order = np.argsort(y_pred)
    y_true = np.take(y_true, order)
    pos = 1 + np.where(y_true > 0)[0]
    n_rels = 1 + np.arange(len(pos))
    return np.mean(n_rels / pos) if len(pos) > 0 else 0



def average_precision(r):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    >>> delta_r = 1. / sum(r)
    >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
    0.7833333333333333
    >>> average_precision(r)
    0.78333333333333333
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Average precision
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
    >>> mean_average_precision(rs)
    0.78333333333333333
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
    >>> mean_average_precision(rs)
    0.39166666666666666
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])


def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)

# a = [3,3,3,2,2,2,1,0]
# pred = [3,2,3,0,1,2]
 
# sorter = NDCGScorer(K=5)
# print(sorter(a,pred))