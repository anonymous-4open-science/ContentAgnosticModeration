import logging
import pickle
logger = logging.getLogger(__name__)

import numpy as np
from omegaconf import DictConfig


def load_data(config: DictConfig):    
    init_matrix_path=config.paths[config.scenario].init_matrix_path
    users_clicked_pool_path=config.paths[config.scenario].users_clicked_pool_path
    users_dict_path=config.paths[config.scenario].users_dict_path
    news_df_path=config.paths[config.scenario].news_df_path            

    init_matrix = pickle.load(open(init_matrix_path, 'rb'))
    users_clicked_pool = pickle.load(open(users_clicked_pool_path, 'rb'))
    users_dict = pickle.load(open(users_dict_path, 'rb'))
    news_df = pickle.load(open(news_df_path, 'rb'))

    return {
        'init_matrix': init_matrix,
        'users_clicked_pool': users_clicked_pool,
        'users_dict': users_dict,
        'news_df': news_df
    }