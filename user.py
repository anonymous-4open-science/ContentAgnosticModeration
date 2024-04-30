# user models
import numpy as np
from numpy import dot
from numpy.linalg import norm

from abc import ABC, abstractmethod

class BaseUser(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def update(self, *args, **kwargs):
        """
        Update the state of the user. Subclasses must implement this method.
        """
        pass

    @abstractmethod
    def interaction(self, *args, **kwargs):
        """
        Define how the user interacts. Subclasses must implement this method.
        """
        pass


class BiasedUser(BaseUser):
    def __init__(self, c=0.03, epsilon=10e-5, preference_beta=0.98): 
        super().__init__()
        self.c = c
        self.epsilon = epsilon
        self.preference_beta = preference_beta
        self.type = 'biased'

    # def update(self, u_vec, i_vec, c=None):
    #     if c is None:
    #         c = self.c
    #     new_vec = u_vec.copy()
    #     new_vec += c * i_vec
    #     return new_vec
    def update(self, u_vec, i_vec_list):
        new_vec = u_vec.copy()
        for i_vec in i_vec_list:
            new_vec += self.c * i_vec
        return new_vec

    def interaction(self, uv, iv):
        """
        Given a user vector (uv) and a recommended new, 
        return whether user is gonna click or not
        """

        # cos_sim = dot(uv, iv)/(norm(uv)*norm(iv))
        # v = [i/sum(uv) for i in iv] #TODO v3
        v = [i/sum(iv) for i in iv]
        # v = iv
        utility = np.dot(uv,v)

        if (utility + self.epsilon) >= 1.0:
            vui = 0.99
        else:
            vui = beta_distribution(utility)

        # Awared preference
        ita = beta_distribution(self.preference_beta)
        # print('vui:', vui, 'ita:', ita)
        pui = vui * ita

        rand_num = np.random.random()

        if rand_num < pui:
        # if rand_num < utility:
            return 1
        else:
            return 0


class OpenUser(BaseUser):
    def __init__(self, c_start=0.03):
        super().__init__()
        self.c = c_start
        self.type = 'open'
    
    def update(self, u_vec, i_vec_list):
        new_vec = u_vec.copy()
        starting_weight = self.c
        # add with logarithm decaying weight, similar with NDCG
        for i_vec in i_vec_list:
            new_vec += starting_weight * i_vec
            starting_weight = starting_weight * 0.9 # TODO 
        return new_vec
   
    def interaction(self, uv, iv): # TODO, read all or random items from topk 
        return 1
        

class RandomUser(OpenUser):
    def __init__(self, c_start=0.03):
        super().__init__(c_start=c_start)
        self.p = 0.5
        self.type = 'random'
   
    def interaction(self, uv, iv): # TODO, read all or random items from topk 
        return np.random.choice([0, 1], p=[self.p, 1-self.p]) # TODO: config prob



def beta_distribution(mu, sigma=10 ** -5):
    """
    Sample from beta distribution given the mean and variance. 
    """
    alpha = mu * mu * ((1 - mu) / (sigma * sigma) - 1 / mu)
    beta = alpha * (1 / mu - 1)

    return np.random.beta(alpha, beta)


def read_proxy(uv, iv):
    cos_sim = np.dot(uv,iv) # the stance and topic entry of the user vector

    # mean_uv = np.mean(uv)
    # # 75 quantile of the user vector
    # lb_uv = np.quantile(uv, 0.75)
    lb_uv = np.median(uv)
    return (cos_sim > lb_uv).astype(int)
    

def user_update(u_vec, i_vec, c):
    new_vec = u_vec.copy()
    new_vec += c * i_vec
    return new_vec