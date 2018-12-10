import math

import numpy as np
import sklearn
from metric import NDCGScorer, NDCGScorer_qid
from sklearn.utils import check_X_y
from weak_ranker import WeakRanker
from copy import deepcopy

class AdaRank():
    # scorer NDCG/MAP
    def __init__(self, scorer=None):
        self.scorer = scorer
        self.used_fids = []
        self.rweight = None
        self.featureQueue = None
        self.weak_rankers = None
        self.sweights = None
        self.last_trained_score_mean = -np.inf
        self.lastFeatureConsecutiveCount = 0
        self.last_fid = -1
        self.__max_selected_count = 5  # the max number of times a feature can be selected consectuively before being removed
        self.tolerance = 0.0001
        self.max_iter = 500
        # dataset
        self.X = None
        self.y = None
        self.qid = None
        self.X_valid = None
        self.y_valid = None
        self.qid_valid = None
        self.backup_trained_score = None
        self.backup_sweights = None
        self.best_valid_score = -1
        self.best_model_rankers = None
        self.best_model_rweight = None

    def fit(self, X, y, qid, X_valid=None, y_valid=None, qid_valid=None):
        # verify the test set
        self.X, self.y = check_X_y(X, y, 'csr')
        self.X = self.X

        # verify the valid
        if X_valid is not None:
            X_valid, y_valid = check_X_y(X_valid, y_valid, 'csr')
            self.X_valid = X_valid
            self.y_valid = y_valid

        # use nDCG@5 as the default scorer
        if self.scorer is None:
            self.scorer = NDCGScorer_qid(K=5)

        self.qid = qid
        self.qid_valid = qid_valid
        # get the number of quries
        self.n_queries = np.unique(self.qid).shape[0]

        # compute the score for different rankers over features
        # initial the weight and normizalied
        # weight size equal to queries size (P)
        self.sweights = np.ones(
            self.n_queries, dtype=np.float64) / self.n_queries

        self.n_features = self.X.shape[1]

        self.init_WR(self.X, self.y, self.qid)
        self.bset_weak_rankers = []

        t = self.learn()


    def learn(self):
        # T decide automatically
        t = 0
        count_fid = {}
        for i in range(self.n_features):
            count_fid[i] = 0

        while (t < self.max_iter):
            
            t += 1
            best_WR = self.learn_weak_ranker()


            if best_WR is None:
                break


            count_fid[best_WR.get_fid()] += 1

            if count_fid[best_WR.get_fid()] > 5:
                self.used_fids.append(best_WR.get_fid())

            alpha = best_WR.cal_alpha(self.sweights)

            assert alpha > 0
            best_WR.set_alpha(alpha)
            self.bset_weak_rankers.append(best_WR)
            self.rweight = self.generate_rweight()


            # score both training and validation data
            # if the performance didn't increased, stop the iteration
            trained_score = self.scorer(self.y, np.dot(self.X, self.rweight),
                                        self.qid)

            trained_score_mean = trained_score.mean()

            delta = trained_score_mean + self.tolerance - self.last_trained_score_mean


            # save best model in validation
            if t% 1 == 0 and self.X_valid is not None:
                valid_score = self.scorer(self.y_valid,
                                                np.dot(self.X_valid, self.rweight),
                                                self.qid_valid).mean()
                if valid_score > self.best_valid_score:
                    self.best_valid_score  = valid_score
                    self.best_model_rankers = self.bset_weak_rankers[:]


            if delta <= 0:
                del self.bset_weak_rankers[-1]
                self.rweight = self.generate_rweight()
                break

            # Update sample weight
            self.last_trained_score_mean = trained_score_mean
            new_sweights = np.exp(-1 * self.scorer(self.y, np.dot(self.X, self.rweight),
                                        self.qid))
            self.sweights = new_sweights / new_sweights.sum()


        return t

    def predict(self, X):
        final_model = self.model()
        return np.dot(X, final_model)

    def generate_rweight(self):
        model = np.zeros(self.n_features)
        for ranker in self.bset_weak_rankers:
            model[ranker.get_fid()] += ranker.get_alpha()
        return model

    def generate_best_rweight(self):
        model = np.zeros(self.n_features)
        for ranker in self.best_model_rankers:
            model[ranker.get_fid()] += ranker.get_alpha()
        return model 

    def model(self):
        return self.generate_best_rweight()

    def learn_weak_ranker(self):
        best_scores = -np.inf
        best_WR = None
        for ranker in self.weak_rankers:
            if ranker.get_fid() in self.used_fids:
                continue

            scores = np.dot(self.sweights, ranker.get_score()).sum()
            if scores > best_scores:
                best_WR = ranker
                best_scores = scores
        return deepcopy(best_WR)

    def init_WR(self, X, y, qid):
        # compute the score for different rankers over features
        # weight will not influence these scores, so we can compute advance
        self.weak_rankers = []
        for j in range(0, self.n_features):
            pred = X[:, j].ravel()
            self.weak_rankers.append(WeakRanker(j, self.scorer(y, pred, qid)))
            