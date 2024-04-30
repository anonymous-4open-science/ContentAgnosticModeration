import numpy as np
import pandas as pd
import pickle
from scipy.stats import entropy as scipy_entropy
from scipy.stats import skew
from itertools import accumulate
import matplotlib.pyplot as plt
from collections import defaultdict
from numpy.linalg import norm
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances

from utils import *


# get distibution of topics and stances
def get_topic_stance_dist(matrix, inner_id_to_stance_topic):
    item_counts = np.sum(matrix, axis=0)
    stance_distribution = np.zeros(3)
    topic_distribution = np.zeros(7)
    for inner_id, count in enumerate(item_counts):
        stance, topic = inner_id_to_stance_topic[inner_id]
        stance_distribution[stance] += count
        topic_distribution[topic] += count
    return stance_distribution, topic_distribution


def calculate_variance(matrix, axis=0):
    """Calculate variance along the specified axis."""
    return np.var(matrix, axis=axis)

def calculate_entropy(matrix, axis=0):
    """Calculate entropy along the specified axis."""
    # Convert to proportions
    proportions = matrix / np.sum(matrix, axis=axis, keepdims=True)
    return scipy_entropy(proportions, base=2, axis=axis)

def calculate_hhi(matrix, axis=0):
    """Calculate the Herfindahl-Hirschman Index (HHI) along the specified axis."""
    # Convert to proportions
    proportions = matrix / np.sum(matrix, axis=axis, keepdims=True)
    # Square the proportions and sum to get HHI
    return np.sum(np.square(proportions), axis=axis)


def calculate_gini_index(matrix, axis=0):
    # Sort the matrix along the specified axis
    sorted_matrix = np.sort(matrix, axis=axis)
    
    # Calculate cumulative sums and normalize them by the total sum
    if axis == 0:
        cumsum = np.cumsum(sorted_matrix, axis=axis) / np.sum(sorted_matrix, axis=axis)
    else:
        # Adjust for axis=1; ensure correct shapes for broadcasting
        cumsum = (np.cumsum(sorted_matrix, axis=axis).T / np.sum(sorted_matrix, axis=axis)).T
    
    # Calculate the Gini index using the normalized cumulative sums
    # This involves integrating the Lorenz curve, which the normalized cumsum approximates
    n = matrix.shape[axis]
    if axis == 0:
        # Direct calculation for each column
        gini = 1 - 2 * np.sum(cumsum, axis=axis) / n + 1/n
    else:
        # Ensure correct calculation for each row
        gini = 1 - 2 * np.sum(cumsum, axis=axis) / n + 1/n
    
    return gini


def inverse_simpson_index(matrix, axis=0):
    """
    Calculate the Inverse Simpson Index for each column (topic) in the matrix
    when axis=0, or for each row (stance) when axis=1.
    """
    N = np.sum(matrix, axis=axis)
    
    if axis == 0:
        reshaped_N = N.reshape(1, -1)  # Correct reshaping for column-wise operation
    elif axis == 1:
        reshaped_N = N.reshape(-1, 1)  # Correct reshaping for row-wise operation

    # Perform the division with correctly reshaped N for broadcasting
    proportion_matrix = matrix / reshaped_N
    squared_proportions = np.square(proportion_matrix)
    sum_squared_proportions = np.sum(squared_proportions, axis=axis)
    
    # Calculate and return the Inverse Simpson Index
    diversity_indices = 1 / sum_squared_proportions
    return diversity_indices


def agged_index(matrix_grouped, index_func):
    agged_index = 0.0
    for u, matrix in matrix_grouped.items():
        agged_index += index_func(matrix).mean()
    return agged_index/len(matrix_grouped)


def indices_all(init_matrix, shown, read, user_to_type, inner_id_to_stance_topic):
    init_matrix_grouped = group_by_user_type_topic_stance(init_matrix, user_to_type, inner_id_to_stance_topic)
    shown_grouped = group_by_user_type_topic_stance(shown, user_to_type, inner_id_to_stance_topic)
    read_grouped = group_by_user_type_topic_stance(read, user_to_type, inner_id_to_stance_topic)
    agged_diversity = agged_index(init_matrix_grouped, inverse_simpson_index)
    agged_diversity_shown = agged_index(shown_grouped, inverse_simpson_index)
    agged_diversity_read = agged_index(read_grouped, inverse_simpson_index)
    print('Diversity index\nInitial: {}, Shown after iterations: {}, Read after iterations {}'.format(agged_diversity, agged_diversity_shown, agged_diversity_read))
    agged_variance = agged_index(init_matrix_grouped, calculate_variance)
    agged_variance_shown = agged_index(shown_grouped, calculate_variance)
    agged_variance_read = agged_index(read_grouped, calculate_variance)
    print('Variance index\nInitial: {}, Shown after iterations: {}, Read after iterations {}'.format(agged_variance, agged_variance_shown, agged_variance_read))
    agged_entropy = agged_index(init_matrix_grouped, calculate_entropy)
    agged_entropy_shown = agged_index(shown_grouped, calculate_entropy)
    agged_entropy_read = agged_index(read_grouped, calculate_entropy)
    print('Entropy index\nInitial: {}, Shown after iterations: {}, Read after iterations {}'.format(agged_entropy, agged_entropy_shown, agged_entropy_read))
    agged_gini = agged_index(init_matrix_grouped, calculate_gini_index)
    agged_gini_shown = agged_index(shown_grouped, calculate_gini_index)
    agged_gini_read = agged_index(read_grouped, calculate_gini_index)
    print('Gini index\nInitial: {}, Shown after iterations: {}, Read after iterations {}'.format(agged_gini, agged_gini_shown, agged_gini_read))
    return agged_diversity, agged_diversity_shown, agged_diversity_read, agged_variance, agged_variance_shown, agged_variance_read, agged_entropy, agged_entropy_shown, agged_entropy_read, agged_gini, agged_gini_shown, agged_gini_read


def bicluster_silhouette_score_z(z, row_labels, col_labels, n_rows=None):
    """
    Calculate a weighted average silhouette score for biclusters using the stacked embeddings matrix `z`.

    Parameters:
    - z: The stacked embeddings matrix from spectral co-clustering.
    - row_labels: The clustering labels for rows.
    - col_labels: The clustering labels for columns.
    - n_rows: The number of rows in the original data matrix (to separate row and column embeddings in `z`).

    Returns:
    - weighted_silhouette_score: The weighted average silhouette score for the biclustering.
    """
    if not n_rows:
        n_rows = len(row_labels)
    # Separate row and column embeddings
    row_embeddings = z[:n_rows]
    col_embeddings = z[n_rows:]
    
    row_distances = euclidean_distances(row_embeddings)
    col_distances = euclidean_distances(col_embeddings)
    
    # Initialize lists to store silhouette scores and cluster sizes
    row_silhouettes = []
    col_silhouettes = []
    row_cluster_sizes = []
    col_cluster_sizes = []
    row_cluster_silhouttes = []
    col_cluster_silhouttes = []

    # Unique labels
    row_clusters = np.unique(row_labels)
    col_clusters = np.unique(col_labels)
    
    # Calculate silhouettes for rows
    for row_cluster in row_clusters:
        cluster_members = (row_labels == row_cluster)
        cluster_size = np.sum(cluster_members)
        row_cluster_sizes.append(cluster_size)
        if cluster_size <= 1:
            continue  # Silhouette is not defined for clusters of size 1
        a_row = np.mean(row_distances[cluster_members][:, cluster_members], axis=1)
        # init cluster size of inf np.array
        b_row = np.full(cluster_size, np.inf)
        # print(b_row.shape)
        temp_results = []
        for other_cluster in row_clusters:
            if other_cluster != row_cluster:
                if np.sum(row_labels == other_cluster) > 0:
                    temp_results.append(np.mean(row_distances[cluster_members][:, row_labels == other_cluster], axis=1))
                else:
                    temp_results.append(b_row)
        b_row = np.min(temp_results, axis=0)
        # b_row = np.min([np.mean(row_distances[cluster_members][:, row_labels == other_cluster], axis=1) 
        #                 for other_cluster in row_clusters if other_cluster != row_cluster], axis=0)
        s_row = (b_row - a_row) / np.maximum(a_row, b_row)
        row_silhouettes.append(s_row)
        row_cluster_silhouttes.append(np.mean(s_row))
    
    # Calculate silhouettes for columns
    for col_cluster in col_clusters:
        cluster_members = (col_labels == col_cluster)
        cluster_size = np.sum(cluster_members)
        col_cluster_sizes.append(cluster_size)
        if cluster_size <= 1:
            continue  # Silhouette is not defined for clusters of size 1
        a_col = np.mean(col_distances[cluster_members][:, cluster_members], axis=1)
        b_col = np.full(cluster_size, np.inf)
        # print(b_col.shape)
        temp_results = []
        for other_cluster in col_clusters:
            if other_cluster != col_cluster:
                if np.sum(col_labels == other_cluster) > 0:
                    temp_results.append(np.mean(col_distances[cluster_members][:, col_labels == other_cluster], axis=1))
                else:
                    temp_results.append(b_col)
        b_col = np.min(temp_results, axis=0)
        # b_col = np.min([np.mean(col_distances[cluster_members][:, col_labels == other_cluster], axis=1) 
        #                 for other_cluster in col_clusters if other_cluster != col_cluster], axis=0)
        s_col = (b_col - a_col) / np.maximum(a_col, b_col)
        col_silhouettes.append(s_col)
        col_cluster_silhouttes.append(np.mean(s_col))
    
    # Weighted average silhouette score
    total_size = sum(row_cluster_sizes + col_cluster_sizes)
    # weighted_silhouette_score = (sum(sum(s) * size for s, size in zip(row_silhouettes, row_cluster_sizes)) + 
    #                              sum(sim)s) * size for s, size in zip(col_silhouettes, col_cluster_sizes))) / total_size
    weighted_silhouette_score = np.mean(row_cluster_silhouttes + col_cluster_silhouttes) #TODO no weights, plain avg
    
    # print(list(zip(row_cluster_silhouttes, row_cluster_sizes)))
    # print(list(zip(col_cluster_silhouttes, col_cluster_sizes)))
    row_results_dict = dict(zip(row_clusters, [(i, j) for (i, j) in zip(row_cluster_silhouttes, row_cluster_sizes)]))
    col_results_dict = dict(zip(col_clusters, [(i, j) for (i, j) in zip(col_cluster_silhouttes, col_cluster_sizes)]))
    
    return weighted_silhouette_score, row_results_dict, col_results_dict

# VISUALIZATION
def plot_stance_topic_distribution(stance_distribution, topic_distribution):
    fig, ax = plt.subplots(1,2, figsize=(12,4))
    ax[0].bar(['left', 'center', 'right'], stance_distribution)
    ax[0].set_title('Stance distribution')
    ax[1].bar(np.arange(7), topic_distribution)
    ax[1].set_title('Topic distribution')
    plt.show()

    
# stance distribution grouped by user stance
def plot_stance_topic_distribution_per_group(user_type, type_to_users, init_matrix, inner_id_to_stance_topic):
    user_list = type_to_users[user_type]
    init_matrix_user = init_matrix[user_list]
    item_counts_user = np.sum(init_matrix_user, axis=0)
    user_stance_distribution = np.zeros(3)
    user_topic_distribution = np.zeros(7)
    for inner_id, count in enumerate(item_counts_user):
        stance, topic = inner_id_to_stance_topic[inner_id]
        user_stance_distribution[stance] += count
        user_topic_distribution[topic] += count
    fig, ax = plt.subplots(1,2, figsize=(12,4))
    ax[0].bar(['left', 'center', 'right'], user_stance_distribution)
    ax[0].set_title('Stance distribution')
    ax[1].bar(np.arange(7), user_topic_distribution)
    ax[1].set_title('Topic distribution')
    # set a common title for the two plots
    fig.suptitle(user_type)
    plt.show()

    
def get_sliced_history(history, user_to_type):
    history_t_user_group = {}
    for t, v in history.items():
        if t == 'read_ratios':
            continue
        new_d = defaultdict(dict)
        for u, v_ in v.items():
            shown = v_['shown']
            read = v_.get('read', [])
            group = user_to_type[u]
            if 'shown' not in new_d[group]:
                new_d[group]['shown'] = []
            if 'read' not in new_d[group]:
                new_d[group]['read'] = []
            new_d[group]['shown'].extend(shown)
            new_d[group]['read'].extend(read)
        history_t_user_group[t] = new_d
    return history_t_user_group

# item counts per time step
def counts_by_time_step(history_t_user_group, g='Progressive Left', n=2100, t_included=100):
    shown_item_counts_per_t = {}
    read_item_counts_per_t = {}
    for t in range(t_included):
        shown_item_counts_per_t[t] = np.zeros(n)
        read_item_counts_per_t[t] = np.zeros(n)

    for t, values in history_t_user_group.items():
        if t >= t_included:
            break
        if g is not None:
            for g_, res in values.items():
                if g_ == g:
                    for i in res['shown']:
                        shown_item_counts_per_t[t][i] += 1
                    for i in res.get('read', []):
                        read_item_counts_per_t[t][i] += 1
        else:
            for res in values.values():
                for i in res['shown']:
                    shown_item_counts_per_t[t][i] += 1
                for i in res.get('read', []):
                    read_item_counts_per_t[t][i] += 1
    return shown_item_counts_per_t, read_item_counts_per_t            

# for each time step, calculate stance and topic distribution
def dist_by_time_step(history_t_user_group,inner_id_to_stance_topic, g='Progressive Left', n=2100, t_included=100):
    shown_item_counts_per_t, read_item_counts_per_t = counts_by_time_step(history_t_user_group, g, n, t_included)
    shown_dist = {}
    read_dist = {}
    for t, item_counts in shown_item_counts_per_t.items():
        stance_distribution = np.zeros(3)
        topic_distribution = np.zeros(7)
        for inner_id, count in enumerate(item_counts):
            stance, topic = inner_id_to_stance_topic[inner_id]
            stance_distribution[stance] += count
            topic_distribution[topic] += count
        shown_dist[t] = (stance_distribution, topic_distribution)

    for t, item_counts in read_item_counts_per_t.items():
        stance_distribution = np.zeros(3)
        topic_distribution = np.zeros(7)
        for inner_id, count in enumerate(item_counts):
            stance, topic = inner_id_to_stance_topic[inner_id]
            stance_distribution[stance] += count
            topic_distribution[topic] += count
        read_dist[t] = (stance_distribution, topic_distribution)
    return shown_dist, read_dist
    
    
# plot t vs stance distribution
def plot_single_group(history_t_user_group, inner_id_to_stance_topic,g='Committed Conservatives', n=2100, t_included=100):
    shown_dist, read_dist = dist_by_time_step(history_t_user_group,inner_id_to_stance_topic, g, n, t_included)
    # plot two subplots
    fig, ax = plt.subplots(1,2, figsize=(12,4))

    plot_lines(ax[0], t_included, read_dist)
    plot_lines(ax[1], t_included, shown_dist, 'Shown')
    # a common title for the two plots
    fig.suptitle(g)
    plt.show()

def plot_lines(ax, t_included, dist, title='Read'):
    xs = np.arange(t_included)
    ys_l = []
    ys_c = []
    ys_r = []
    for v in dist.values():
        ys_l.append(v[0][0])
        ys_c.append(v[0][1])
        ys_r.append(v[0][2])
    # accumulate ys_l
    ys_l_acc = list(accumulate(ys_l))
    ys_c_acc = list(accumulate(ys_c))
    ys_r_acc = list(accumulate(ys_r))

    ax.plot(xs, ys_l_acc, label='L')
    ax.plot(xs, ys_c_acc, label='C')
    ax.plot(xs, ys_r_acc, label='R')

    #legend
    ax.legend()
    ax.set_title(title)

    
def plot_all_groups(history_t_user_group,inner_id_to_stance_topic, n=2100, t_included=100):
    shown_dist, read_dist = dist_by_time_step(history_t_user_group, inner_id_to_stance_topic, g=None, n=n, t_included=t_included)
    # plot two subplots
    fig, ax = plt.subplots(1,2, figsize=(12,4))

    plot_lines(ax[0], t_included, read_dist)
    plot_lines(ax[1], t_included, shown_dist, 'Shown')
    # a common title for the two plots
    fig.suptitle('All groups Aggregated')
    plt.show()
    
# interaction matrix after t steps
def interaction_matrix_upto_t(init_matrix, history, t_included=100):
    shown = init_matrix.copy()
    read = init_matrix.copy()
    for t, v in history.items():
        if t == 'read_ratios':
            continue
        if t >= t_included:
            break
        for u, sr in v.items():
            for i in sr['shown']:
                shown[u, i] +=1
            for i in sr.get('read', []):
                read[u, i] += 1
    # plot the two matrices
    
    return shown, read
#
# item stance and topic distribution of shown matrix
def plot_all_group_bars(matrix, inner_id_to_stance_topic, title='Shown'):
    item_counts = np.sum(matrix, axis=0)
    len(item_counts)
    stance_distribution = np.zeros(3)
    topic_distribution = np.zeros(7)
    for inner_id, count in enumerate(item_counts):
        stance, topic = inner_id_to_stance_topic[inner_id]
        stance_distribution[stance] += count
        topic_distribution[topic] += count
    # plot two subplots
    fig, ax = plt.subplots(1,2, figsize=(12,4))

    ax[0].bar(['left', 'center', 'right'], stance_distribution)
    ax[0].set_title('Stance distribution')

    # plot topic distribution
    ax[1].bar(range(7), topic_distribution)
    ax[1].set_title('Topic distribution')

    # a common title for the two plots
    fig.suptitle(title)
    plt.show()

# item stance per user group
def plot_single_group_bars(user_group, type_to_users, inner_id_to_stance_topic, matrix, title='Shown'):
    user_list = type_to_users[user_group]
    matrix_user = matrix[user_list]
    item_counts_user = np.sum(matrix_user, axis=0)
    user_stance_distribution = np.zeros(3)
    user_topic_distribution = np.zeros(7)
    for inner_id, count in enumerate(item_counts_user):
        stance, topic = inner_id_to_stance_topic[inner_id]
        user_stance_distribution[stance] += count
        user_topic_distribution[topic] += count
    # plot two subplots
    fig, ax = plt.subplots(1,2, figsize=(12,4))
    ax[0].bar(['left', 'center', 'right'], user_stance_distribution)
    ax[0].set_title('Stance distribution')
    ax[1].bar(range(7), user_topic_distribution)
    ax[1].set_title('Topic distribution')
    # set a common title for the two plots
    fig.suptitle(user_group + ' (' + title + ')')
    plt.show()

def get_uv_before_after(users_dict, users_dict_after):

    ptts = set([v['ptt'] for v in users_dict.values()])
    ptts_stance = {}
    ptts_stance_after = {}
    for ptt in ptts:
        ptts_stance[ptt] = []
        ptts_stance_after[ptt] = []

    for u, v in users_dict.items():
        ptts_stance[v['ptt']].append(v['vec'])        
        ptts_stance_after[users_dict_after[u]['ptt']].append(users_dict_after[u]['vec'])
    vec_dim = len(v['vec'])
    print(vec_dim)
    
    ptts_stance_mean = {}
    for ptt, stances in ptts_stance.items():
        ptts_stance_mean[ptt] = np.mean(stances, axis=0)

    ptts_stance_mean_after = {}
    for ptt, stances in ptts_stance_after.items():
        ptts_stance_mean_after[ptt] = np.mean(stances, axis=0)

    stance_mask = [0,1,2] * (vec_dim//3)
    stance_mask = np.array(stance_mask)
    print(stance_mask.shape)
    
    ptts_stance_agg = {}
    for k, v in ptts_stance_mean.items():
        ptts_stance_agg[k] = np.dot(v, stance_mask-1)

    ptts_stance_agg_after = {}
    for k, v in ptts_stance_mean_after.items():
        print(v.shape, stance_mask.shape)
        ptts_stance_agg_after[k] = np.dot(v, stance_mask-1)

    return ptts_stance, ptts_stance_after, ptts_stance_agg, ptts_stance_agg_after

def plot_uv_delta(ptts_stance_agg, ptts_stance_agg_after):
    plt.bar(ptts_stance_agg.keys(), ptts_stance_agg.values())
    plt.bar(ptts_stance_agg_after.keys(), ptts_stance_agg_after.values(), alpha=0.5)
    plt.xticks(rotation=90)
    plt.show()

# lorenz curve  
def lorenz_curve(data):
    data = np.array(data)
    data = data/np.sum(data)
    data = np.sort(data)
    n = len(data)
    x = np.linspace(0, 1, n)
    y = np.cumsum(data)
    return x, y


def plot_lorenz_curves(init_matrix, after_matrix, label='Shown'):
    before = init_matrix.sum(axis=0)
    after = after_matrix.sum(axis=0)
    x, y = lorenz_curve(before)
    plt.plot(x, y, label='Time 0')
    x, y = lorenz_curve(after)
    plt.plot(x, y, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend()
    plt.show()


# def get_skewness(init_matrix, inner_id_to_stance_topic, history, t=None):
#     if not t:
#         t = len(history)-1
#     else:
#         t = min(t, len(history)-1)
#     shown, read = interaction_matrix_upto_t(init_matrix, history, t)
#     stance, topic =  get_topic_stance_dist(shown, inner_id_to_stance_topic)
#     # print('\t\t\t\t\t\t', stance, topic)
#     data2 = [-1]* int(stance[0]) + [0]* int(stance[1]) + [1] * int(stance[2])
#     skew_shown = skew(data2)
#     # print(skew_shown)
#     stance, topic =  get_topic_stance_dist(read, inner_id_to_stance_topic)
#     # print('\t\t\t\t\t\t', stance, topic)
#     data3 = [-1]* int(stance[0]) + [0]* int(stance[1]) + [1] * int(stance[2])
#     skew_read = skew(data3)
#     # print(skew_read)
#     return skew_shown, skew_read


def plots_all(history, init_matrix, inner_id_to_stance_topic, user_to_type, type_to_users, users_dict, users_dict_after, plot_init=True):
    m, n = init_matrix.shape
    stance_distribution, topic_distribution = get_topic_stance_dist(init_matrix, inner_id_to_stance_topic)
    if plot_init:
        print('Initial stance and topic distribution')
        plot_stance_topic_distribution(stance_distribution, topic_distribution)    

        for k in type_to_users.keys():
            plot_stance_topic_distribution_per_group(k, type_to_users, init_matrix, inner_id_to_stance_topic)

    print('Recommendation by time step')
    history_t_user_group = get_sliced_history(history, user_to_type)
    plot_all_groups(history_t_user_group, inner_id_to_stance_topic, n=n, t_included=len(history)-1)
    for g in history_t_user_group[0].keys():
        plot_single_group(history_t_user_group, inner_id_to_stance_topic, g=g, n=n, t_included=len(history)-1)

    # --------------------------------------
    print('\nAfter {} steps'.format(len(history)))
    shown_dict = {}
    read_dict = {}

    shown, read = interaction_matrix_upto_t(init_matrix, history, len(history)-1)
    shown_dict[len(history)] = shown
    read_dict[len(history)] = read

    plot_all_group_bars(shown, inner_id_to_stance_topic, 'Shown')
    plot_all_group_bars(read, inner_id_to_stance_topic, 'Read')
    for k in type_to_users.keys():
        plot_single_group_bars(k, type_to_users, inner_id_to_stance_topic, shown)
    for k in type_to_users.keys():
        plot_single_group_bars(k, type_to_users, inner_id_to_stance_topic, read, title='Read')
    
    # --------------------------------------
    # print('\nAfter 20 steps')
    # shown, read = interaction_matrix_upto_t(init_matrix, history, 20)
    # shown_dict[20] = shown
    # read_dict[20] = read

    # plot_all_group_bars(shown, inner_id_to_stance_topic, 'Shown')
    # plot_all_group_bars(read, inner_id_to_stance_topic, 'Read')
    # for k in type_to_users.keys():
    #     plot_single_group_bars(k, type_to_users, inner_id_to_stance_topic, shown)
    # for k in type_to_users.keys():
    #     plot_single_group_bars(k, type_to_users, inner_id_to_stance_topic, read, title='Read')

    print('User vector after {} steps'.format(len(history)))
    ptts_stance, ptts_stance_after, ptts_stance_agg, ptts_stance_agg_after = get_uv_before_after(users_dict, users_dict_after)
    plot_uv_delta(ptts_stance_agg, ptts_stance_agg_after)
    
    # print('Lorenz curve after 20 steps')
    # plot_lorenz_curves(init_matrix, shown_dict[20], 'Shown')
    # plot_lorenz_curves(init_matrix, read_dict[20], 'Read')
    print('Lorenz curve after {} steps'.format(len(history)))
    plot_lorenz_curves(init_matrix, shown, 'Shown')
    plot_lorenz_curves(init_matrix, read, 'Read')
    return shown_dict, read_dict



    

