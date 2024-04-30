import pickle
import torch
import numpy as np
import pandas as pd
from torchnmf.nmf import NMF

import torch.nn as nn
import torch.optim as optim

from collections import defaultdict

from numpy import dot
from numpy.linalg import norm

class Recommender:
    pass

# recommend top 5 articles for each user
def recommend_topk(indices, users_clicked_pool, k=5):
    """
    Recommend top k articles for each user
    """
    # print("Recommend top {} articles for each user".format(k))
    res = {}
    for i in range(indices.shape[0]):
        idx = indices[i]
        # check if the article is clicked by the user
        clicked = users_clicked_pool[i]
        rec = []
        for j in idx:
            if j not in clicked:
                rec.append(j)
            if len(rec) == k:
                break
        res[i] = rec
    return res


class NmfRecommender:
    def __init__(self, m, n, rank, n_iter=100, tol=1e-6, seed=0):
        self.m = m
        self.n = n
        self.n_iter = n_iter
        self.tol = tol
        self.rank = rank
        self.model = NMF((self.n, self.m), rank=self.rank)
        self.name = 'nmf'

    def fit(self, matrix, verbose=False):
        matrix = torch.tensor(matrix, dtype=torch.float32)  # input shape m x n
        assert(matrix.shape[0] == self.m)
        #init_matrix.t(), max_iter=100, tol=1e-6, verbose=False
        self.model.fit(matrix.t(), max_iter=self.n_iter, tol=self.tol, verbose=verbose)

    def predict(self, users_clicked_pool, k=5):
        WH = self.model.W @ self.model.H.t()
        indices = np.argsort(WH.detach().numpy(), axis=1)[:,::-1]
        return recommend_topk(indices, users_clicked_pool, k=k)

    def save(self, path):
        torch.save(self.model, path)

    def load(self, path):
        self.model = torch.load(path)



class PopRecommender:
    def __init__(self, m, n):
        self.m = m
        self.n = n
        self.popularity = np.zeros(n)
        self.name = 'pop'

    def fit(self, matrix):
        self.popularity = matrix.sum(axis=0)

    def predict(self, users_clicked_pool, k=5):
        popularity_series = pd.Series(self.popularity)
        popularity_series = popularity_series.sort_values(ascending=False)
        indices = popularity_series.index
        # make m copies
        indices = np.tile(indices, (self.m, 1))
        return recommend_topk(indices, users_clicked_pool, k=k)
    

class OracleRecommender:
    def __init__(self, m, n, w, rep=False):
        self.m = m
        self.n = n
        self.w = w # weights for controling randomness
        self.name = 'oracle'+str(w)
        self.rep = rep

    def gold_click_prob(self, uv, iv):
        """
        Calculate the probability of a click given user vectors and item vectors, compatible with
        both 1D and 2D arrays for user and item vectors.
        
        Parameters:
        - uv: User vectors, a 2D numpy array of shape (m, d) or a 1D array of shape (d,).
        - iv: Item vectors, a 2D numpy array of shape (n, d) or a 1D array of shape (d,).
        
        Returns:
        - A 2D numpy array of shape (m, n) containing the probabilities of clicks if both inputs are 2D,
        a 1D array if one of the inputs is 1D, representing probabilities for each pair.
        """
        # Ensure uv is 2D
        if uv.ndim == 1:
            uv = uv[np.newaxis, :]
        # Ensure iv is 2D
        if iv.ndim == 1:
            iv = iv[np.newaxis, :]
        
        # Normalize iv along the last dimension (d)
        iv_n = iv / iv.sum(axis=1, keepdims=True)
        
        # Calculate dot product using broadcasting, output shape will be (m, n)
        dp = np.dot(uv, iv_n.T)
        
        epsilon = 10e-5
        
        # Use np.where for vectorized conditional operation
        vui = np.where((dp + epsilon) > 1.0, 0.99, dp)
        
        ita = 0.98  # The mean of the beta
        
        # Calculate final probabilities, still in a vectorized manner
        pui = vui * ita
        
        # If either input was 1D, return a 1D array
        if uv.shape[0] == 1 or iv.shape[0] == 1:
            return pui.flatten()
        else:
            return pui

    def fit(self, user_vectors, item_vectors):
        # calculate the gold_click_probs
        # sample with the probability
        gold = self.gold_click_prob(user_vectors, item_vectors)
        weights = np.exp(self.w * gold)
        weights = weights / weights.sum(axis=1, keepdims=True)
        self.weights = weights

    def predict(self, users_clicked_pool, k=5):
        topk = {}
        for i in range(self.m):
            if self.rep:
                idx = np.random.choice(self.n, p=self.weights[i], replace=False, size=k) 
            else:
                idx = []
                while len(idx) < k:
                    candidate = np.random.choice(self.n, p=self.weights[i])
                    if candidate not in users_clicked_pool[i]:
                        idx.append(candidate)
            topk[i] = idx

        return topk