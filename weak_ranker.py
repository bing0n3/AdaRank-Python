import numpy as np


class WeakRanker:
    def __init__(self, fid, score):
        self.__fid = fid
        self.__score = score

    def set_alpha(self, alpha):
        self.__alpha = alpha

    def get_score(self):
        return self.__score

    def cal_alpha(self, weights):
        return 0.5 * np.log(np.dot(weights, 1 + self.__score).sum() / np.dot(weights, 1 - self.__score).sum())

    def get_alpha(self):
        return self.__alpha

    def get_fid(self):
        return self.__fid