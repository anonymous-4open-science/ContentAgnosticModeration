import importlib
import random
from collections import defaultdict

import numpy as np
import torch


def dynamic_import_and_instantiate(module_name: str, class_name: str, params: dict):
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    instance = class_(**params)
    return instance


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

    # all other seeds
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def create_binary_matrix(recommender_output, num_users, num_items):
    binary_matrix = np.zeros((num_users, num_items))
    for user_id, items in recommender_output.items():
        binary_matrix[user_id, items] = 1
    return binary_matrix


def create_real_valued_matrix(recommender_output, num_users, num_items, min_score=0.1, max_score=1.0, bound=0.6):
    real_valued_matrix = np.full((num_users, num_items), min_score)  # Initialize with 0.1
    # score_decrement = (max_score - bound) / (len(next(iter(recommender_output.values()))) - 1)
    for user_id, items in recommender_output.items():
        scores = np.linspace(max_score, bound, len(items))
        real_valued_matrix[user_id, items] = scores
    return real_valued_matrix


def convert_history_to_matrix(history, m, n, key_='read'):
    matrix = np.zeros((m, n))
    for t, user_dict in history.items():
        for user, items in user_dict.items():
            for item in items.get(key_, []):
                matrix[user, item] += 1
    return matrix

def matrix_to_topk_dict(matrix, k):
    topk_dict = {}
    for user_id, scores in enumerate(matrix):
        topk_items = np.argsort(scores)[-k:][::-1]
        topk_dict[user_id] = topk_items.tolist()
    return topk_dict

def defaultdict_list():
    return defaultdict(list)

# Function to recreate the defaultdict structure with the named factory function
def recreate_defaultdict(original, factory_func):
    recreated = defaultdict(factory_func)
    for k, v in original.items():
        if isinstance(v, dict):  # Assuming only one level of nesting for simplicity
            recreated[k] = defaultdict(factory_func, v)
        else:
            recreated[k] = v
    return recreated