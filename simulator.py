import os
import logging
import numpy as np
import pickle
from collections import defaultdict
logger = logging.getLogger(__name__)

from refactoring.moderator import moderate
from refactoring.utils import (
    create_binary_matrix, create_real_valued_matrix, convert_history_to_matrix, matrix_to_topk_dict, dynamic_import_and_instantiate, 
    recreate_defaultdict, defaultdict_list, set_seed
)


class Simulator:
    def __init__(self, config, data_dict):
        self.config = config
        self.data_dict = data_dict 

        logger.info('Set Seed: {}'.format(config.simulation.seed))
        set_seed(config.simulation.seed)        
        
        
        logger.info('Initializing recommender...')
        rec_model_cfg = config.rec_model    
        config.rec_model.params[config.rec_model.type]['m'] = data_dict['init_matrix'].shape[0]
        config.rec_model.params[config.rec_model.type]['n'] = data_dict['init_matrix'].shape[1]
        self.rec_model = dynamic_import_and_instantiate('recommender', rec_model_cfg.type.capitalize() + 'Recommender', rec_model_cfg.params[rec_model_cfg.type])
        
        logger.info('Initializing user model...')
        user_model_cfg = config.user_model
        self.user_model = dynamic_import_and_instantiate('user', user_model_cfg.type.capitalize() + 'User', user_model_cfg.params[user_model_cfg.type])

        logger.info('Set up save directory...')
        save_subdir = config.io.paths[config.io.scenario].save_subdir + '_k{}_seed{}'.format(config.simulation.k, config.simulation.seed)
        self.save_dir = 'results/{}'.format(save_subdir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)

    
    def run(self):
        


        """
        Run simulation for total_time_steps
        """
        # set parameters
        init_matrix, users_dict, users_clicked_pool, news_df = self.data_dict['init_matrix'], self.data_dict['users_dict'], self.data_dict['users_clicked_pool'], self.data_dict['news_df']
        moderator = self.config.moderator.type        

        # simulation
        total_time_steps = self.config.simulation.total_time_steps
        verbose = self.config.simulation.verbose
        k = self.config.simulation.k
        time_limit = self.config.simulation.time_limit

        # recommender
        update_mode = self.config.rec_model.update_mode

        # user
        user_update = self.config.user_model.update_vec

        # moderator
        alpha = self.config.moderator.alpha
        sil_threshold = self.config.moderator.sil_threshold

        print('Moderator: ', moderator)

        # get item vectors for oracle recommender
        item_vectors = np.vstack(news_df['topic_bias_matrix'].values.tolist())

        # history
        history = defaultdict(lambda: defaultdict(dict))
        # create new key in users_dict
        for idx, user in users_dict.items():
            user['vec_over_time'] = {}
            user['vec_over_time'][0] = user['vec'].copy()

        read_ratio_history = []
        # for n timesteps
        for t in range(total_time_steps):
            if verbose:
                print(f'Running simulation for time step {t}')

            # model update
            if update_mode == 'time':
                if 'oracle' not in self.rec_model.name:
                    self.rec_model.fit(init_matrix) 
                else:
                    # fit with current user vector and item vector
                    user_vectors = np.array([u['vec_over_time'][t] for _, u in users_dict.items()])
                    self.rec_model.fit(user_vectors, item_vectors)
            # model predict
            # print(k)
            topk_recommendation = self.rec_model.predict(users_clicked_pool, k=k) 
            # print('Original topk: ', topk)
            if moderator:
                m, n = init_matrix.shape
                # recommendation_matrix = create_binary_matrix(topk, m, n) if moderator == 'popularity_suppression' else create_real_valued_matrix(topk, m, n)
                recommendation_matrix = create_binary_matrix(topk_recommendation, m, n) if moderator == 'mip' else create_real_valued_matrix(topk_recommendation, m, n)
                print('Recommendation matrix shape: ', recommendation_matrix.shape)
                print('Recommendation matrix val: ', recommendation_matrix[0])
                read_history = convert_history_to_matrix(history, m, n, key_='read')
                shown_history = init_matrix + convert_history_to_matrix(history, m, n, key_='shown') # for spectral coclustering           
                
                result = moderate(moderator, recommendation_matrix, topk_recommendation, read_history=read_history, shown_history=shown_history, alpha=alpha, k=k, time_limit=time_limit, sil_threshold=sil_threshold)

                # print('After moderation: ', result)
                if result is None:
                    print('Time step {}: No solution found, return original recommendation\n=========================='.format(t))
                    result = topk_recommendation
                # compare with original recommendation
                # convert topk to matrix
                topk_bi = create_binary_matrix(topk_recommendation, m, n)
                result_bi = create_binary_matrix(result, m, n) if moderator != 'mip' else result
                # print('Diff: ', np.sum(abs(topk_bi - result_bi), axis=1))
                non_zeros = np.where(np.sum(abs(topk_bi - result_bi), axis=0)!=0)[0]
                print('Diff: ', np.sum(abs(topk_bi - result_bi), axis=0)[non_zeros], len(non_zeros))
                print('Diff: ', np.sum(abs(topk_bi - result_bi)))
                topk_recommendation = matrix_to_topk_dict(result, k) if moderator == 'mip' else result

            read_sum = 0

            for idx, user in users_dict.items():
                u_vec = user['vec']
                rec_u = topk_recommendation[idx]
                new_u_vec = u_vec.copy()
                read_i_vec = []
                for i in rec_u:
                    # users_clicked_pool[idx].add(i) # rename exposed pool # TODO: update only the read items???
                    i_vec = news_df[news_df.inner_id == i]['topic_bias_matrix'].values[0].flatten()

                    read = self.user_model.interaction(u_vec, i_vec)

                    if self.user_model.type == 'biased':
                        read_sum += read
                    # else:
                    #     read_sum += read_proxy(user['vec_over_time'][0], i_vec) # if user is not biased, use time 0 vec
                
                    if read:
                        if 'read' not in history[t][idx]:
                            history[t][idx]['read'] = []
                        history[t][idx]['read'].append(i)
                        init_matrix[idx, i] = 1 # update interaction matrix (<---- user model interface)
                        read_i_vec.append(i_vec)
                        users_clicked_pool[idx].add(i) # update clicked pool

                    if 'shown' not in history[t][idx]:
                        history[t][idx]['shown'] = []
                    history[t][idx]['shown'].append(i) # update history matrix
                
                if update_mode == 'user':
                    self.rec_model.fit(init_matrix)
                    
                # update user vector
                if user_update and read_i_vec:
                    # print('Updating user vec...')
                    new_u_vec = self.user_model.update(new_u_vec, read_i_vec)

                user['vec'] = new_u_vec
                user['vec_over_time'][t+1] = new_u_vec 
            # check saturation: if all related items are shown, users will stop reading
            # ratio of read items to total shown items
            read_ratio = read_sum / (len(users_dict) * k)
            read_ratio_history.append(read_ratio)
            print('Read ratio at time {}: '.format(t), read_ratio)
            # if t > 20 and read_ratio < ratio_lb: # 80% users stop reading when k = 5 #TODO run till end
            #     print('Read ratio is less than {}, stop simulation at time step {}'.format(ratio_lb, t))
            #     break

        history['read_ratios'] = read_ratio_history

        # Save history
        history = recreate_defaultdict(history, defaultdict_list)
        user_model_cfg = self.config.user_model
        moderator_cfg = self.config.moderator
        pickle.dump(history, open('{}/history_{}_{}_{}_{}_{}.pkl'.format(self.save_dir, self.rec_model.name, user_model_cfg.type, user_model_cfg.update_vec, moderator_cfg.type, moderator_cfg.alpha), 'wb'))

        # Save users_dict
        pickle.dump(users_dict, open('{}/users_dict_{}_{}_{}_{}_{}.pkl'.format(self.save_dir, self.rec_model.name, user_model_cfg.type, user_model_cfg.update_vec, moderator_cfg.type, moderator_cfg.alpha), 'wb'))

        return history, users_dict