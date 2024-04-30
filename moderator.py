import math
import numpy as np
import skfuzzy as fuzz
from pyscipopt import Model, quicksum
from metrics import bicluster_silhouette_score_z
from sklearn.cluster import SpectralCoclustering
from sklearn.cluster import *
from sklearn.utils.extmath import make_nonnegative
from scipy.linalg import norm
from scipy.sparse import dia_matrix, issparse

def _scale_normalize(X):
    """Normalize ``X`` by scaling rows and columns independently.

    Returns the normalized matrix and the row and column scaling
    factors.
    """
    X = make_nonnegative(X)
    row_diag = np.asarray(1.0 / np.sqrt(X.sum(axis=1))).squeeze()
    col_diag = np.asarray(1.0 / np.sqrt(X.sum(axis=0))).squeeze()
    row_diag = np.where(np.isnan(row_diag), 0, row_diag)
    col_diag = np.where(np.isnan(col_diag), 0, col_diag)
    if issparse(X):
        n_rows, n_cols = X.shape
        r = dia_matrix((row_diag, [0]), shape=(n_rows, n_rows))
        c = dia_matrix((col_diag, [0]), shape=(n_cols, n_cols))
        an = r * X * c
    else:
        an = row_diag[:, np.newaxis] * X * col_diag
    return an, row_diag, col_diag


def _bistochastic_normalize(X, max_iter=1000, tol=1e-5):
    """Normalize rows and columns of ``X`` simultaneously so that all
    rows sum to one constant and all columns sum to a different
    constant.
    """
    # According to paper, this can also be done more efficiently with
    # deviation reduction and balancing algorithms.
    X = make_nonnegative(X)
    X_scaled = X
    for _ in range(max_iter):
        X_new, _, _ = _scale_normalize(X_scaled)
        if issparse(X):
            dist = norm(X_scaled.data - X.data)
        else:
            dist = norm(X_scaled - X_new)
        X_scaled = X_new
        if dist is not None and dist < tol:
            break
    return X_scaled


def _log_normalize(X):
    """Normalize ``X`` according to Kluger's log-interactions scheme."""
    X = make_nonnegative(X, min_value=1)
    if issparse(X):
        raise ValueError(
            "Cannot compute log of a sparse matrix,"
            " because log(x) diverges to -infinity as x"
            " goes to 0."
        )
    L = np.log(X)
    row_avg = L.mean(axis=1)[:, np.newaxis]
    col_avg = L.mean(axis=0)
    avg = L.mean()
    return L - row_avg - col_avg + avg


def apply_fuzzy_c_means(data, n_clusters=3, error=0.005, maxiter=1000):
    """
    Applies Fuzzy C-Means clustering to data.

    Parameters:
    - data: A 2D NumPy array to cluster.
    - n_clusters: Number of clusters.
    - error: Stopping criterion; clusters are considered stable if their change is less than this error.
    - maxiter: Maximum number of iterations.

    Returns:
    - cntr: Cluster centers.
    - u: Updated membership matrix.
    - u0: Initial membership matrix.
    - d: Final distances to cluster centers.
    - jm: Objective function history.
    - p: Number of iterations run.
    - fpc: Fuzzy partition coefficient.
    """
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data, n_clusters, 2, error=error, maxiter=maxiter, init=None)
    return cntr, u, u0, d, jm, p, fpc


class ExtendedSpectralCoclustering(SpectralCoclustering):
    def __init__(self, n_clusters=3, *, svd_method="randomized", n_svd_vecs=None, mini_batch=False, init="k-means++", n_init=10, random_state=None):
        super().__init__(n_clusters=n_clusters, svd_method=svd_method, n_svd_vecs=n_svd_vecs, mini_batch=mini_batch, init=init, n_init=n_init, random_state=random_state)
        self.z_ = None  # Attribute to store the stacked embeddings
        self.u_embeddings_ = None
        self.v_embeddings_ = None

    def _fit(self, X):
        normalized_data, row_diag, col_diag = _scale_normalize(X)
        # n_sv = 1 + int(np.ceil(np.log2(self.n_clusters)))
        u, v = self._svd(normalized_data, 4, n_discard=1)
        z = np.vstack((row_diag[:, np.newaxis] * u, col_diag[:, np.newaxis] * v))

        self.z_ = z
        self.u_embeddings_ = u
        self.v_embeddings_ = v

        centroids, labels = self._k_means(z, self.n_clusters)

        self.centroids_ = centroids

        n_rows = X.shape[0]
        self.row_labels_ = labels[:n_rows]
        self.column_labels_ = labels[n_rows:]

        self.rows_ = np.vstack([self.row_labels_ == c for c in range(self.n_clusters)])
        self.columns_ = np.vstack(
            [self.column_labels_ == c for c in range(self.n_clusters)]
        )      


class FuzzySpectralCoclustering(SpectralCoclustering):
    def __init__(self, n_clusters=3, *, svd_method="randomized", n_svd_vecs=None, mini_batch=False, init="k-means++", n_init=10, random_state=None):
        super().__init__(n_clusters=n_clusters, svd_method=svd_method, n_svd_vecs=n_svd_vecs, mini_batch=mini_batch, init=init, n_init=n_init, random_state=random_state)
        
        self.z_ = None  # Attribute to store the stacked embeddings
        self.u_embeddings_ = None
        self.v_embeddings_ = None
    
    def _fit(self, X):
        # Call the original _fit method to get the normalized data and the svd
        super()._fit(X)
        
        normalized_data, row_diag, col_diag = _scale_normalize(X)
        # n_sv = 1 + int(np.ceil(np.log2(self.n_clusters)))
        n_sv = 4
        u, v = self._svd(normalized_data, n_sv, n_discard=1)
        
        # Combine u and v with row and column weights as done in Spectral Co-clustering
        z = np.vstack((row_diag[:, np.newaxis] * u, col_diag[:, np.newaxis] * v))

        self.z_ = z
        self.u_embeddings_ = u
        self.v_embeddings_ = v
        
        # Instead of _k_means, use fuzzy_c_means to get the membership matrix
        # https://github.com/scikit-fuzzy/scikit-fuzzy/blob/d7551b649f34c2f5e98836e9b502279226d3b225/skfuzzy/cluster/_cmeans.py#L94
        cntr, u, u0, d, jm, p, fpc = apply_fuzzy_c_means(z.T, self.n_clusters) # fuzzy assumes n_feature x n_data as input data
        self.cntr = cntr

        self.membership_matrix_ = u.T
        
        # calculate row_labels_ and column_labels_
        # from the highest membership values
        n_rows = X.shape[0]
        self.row_labels_ = np.argmax(self.membership_matrix_[:n_rows], axis=1)
        self.column_labels_ = np.argmax(self.membership_matrix_[n_rows:], axis=1)

        
def popularity_suppression_linear(recommendation_matrix, alpha1=None, k=5, exposure_history=None, time_limit=600, options=None):
    # Number of users and items
    m, n = recommendation_matrix.shape
    
    # Calculate item popularity weights
    w = np.sum(recommendation_matrix, axis=0) / np.sum(recommendation_matrix)

    if alpha1 is None:
        alpha1 = 0.8

    alpha = (recommendation_matrix * w).sum() * alpha1
    
    model = Model("PopularitySuppressionLinear")
    model.setRealParam('limits/time', time_limit)
    
    # Add binary variables for X
    X = {}
    for i in range(m):
        for j in range(n):
            X[i, j] = model.addVar(vtype="B", name="X_%s_%s" % (i, j))
            
    # Add auxiliary variables for the absolute differences
    AbsDiff = {}
    for i in range(m):
        for j in range(n):
            AbsDiff[i, j] = model.addVar(vtype="C", name="AbsDiff_%s_%s" % (i, j))

    # Minimize the sum of the absolute differences
    model.setObjective(quicksum(AbsDiff[i, j] for i in range(m) for j in range(n)), "minimize")
    
    # Constraints to define the absolute differences
    for i in range(m):
        for j in range(n):
            model.addCons(AbsDiff[i, j] >= recommendation_matrix[i, j] - X[i, j])
            model.addCons(AbsDiff[i, j] >= X[i, j] - recommendation_matrix[i, j])
    
    # Constraint: Aggregated item popularity not over alpha
    model.addCons(quicksum(X[i, j] * w[j] for i in range(m) for j in range(n)) <= alpha, "popularity_constraint")
    
    # Additional constraints to ensure row sums of X match row sums of R
    for i in range(m):
        model.addCons(quicksum(X[i, j] for j in range(n)) == k, "row_sum_constraint_%s" % i)
    
    # Solve the problem
    model.optimize()

    # Check if a solution was found
    sol = model.getBestSol()
    if model.getStatus() == "optimal":
        print("Optimal solution found.")
    elif model.getStatus() == "time_limit":
        print("Reached time limit. Best solution found will be returned.")
    else:
        print("Optimization was stopped with status:", model.getStatus())
    
    # Extract the optimal X matrix
    if sol:
        X_optimized = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                X_optimized[i, j] = model.getVal(X[i, j])
        print(np.sum(X_optimized * w))
        return X_optimized
    else:
        print("No solution found.")
        return None



def greedy_round_robin(num_users, num_items, total_rounds, copies_per_item, relevance_matrix, users, feasible_sets, top_k, exposure_history=None): 
    """
    Perform a greedy round-robin allocation of items to users based on relevance and item availability.
    
    Args:
    - num_users (int): Number of users.
    - num_items (int): Number of items.
    - total_rounds (int): Total rounds of allocation.
    - copies_per_item (int): Initial copies of each item available for allocation.
    - total_copies_to_allocate (int): Total number of item copies to be allocated across all users.
    - relevance_matrix (np.ndarray): Matrix of relevance scores between users and items.
    - users (list): List of user IDs.
    - feasible_sets (dict): Dictionary mapping user IDs to sets of feasible item IDs for allocation.
    - exposure_history (np.ndarray, optional): Matrix indicating the historical exposure of items to users.
    """
    # Initialize allocations for each user
    allocations = {user: [] for user in users}
    
    # Calculate initial item availability, adjusting for historical exposure if provided
    item_availability = {item: copies_per_item - np.sum(exposure_history[:, item]) for item in range(num_items)} if exposure_history is not None else {item: copies_per_item for item in range(num_items)}
    
    # Round-robin allocation process
    for current_round in range(1, total_rounds + 1):
        for user_index in range(num_users):
            if len(allocations[user_index]) == top_k:
                continue
            current_user = users[user_index]
            
            # Identify feasible and available items for the current user
            item_scores = [(item_availability[item] > 0) * (item in feasible_sets[current_user]) * relevance_matrix[current_user, item] for item in range(num_items)]
            
            if any(item_scores):
                selected_item = np.argmax(item_scores)
                
                if item_availability[selected_item] > 0:
                    allocations[current_user].append(selected_item)
                    feasible_sets[current_user].remove(selected_item)
                    item_availability[selected_item] -= 1

    return allocations, feasible_sets


def fair_rec_allocation(recommendation_matrix, exposure_history=None, allocation_limit=1, top_k=5):
    """
    Allocate items to users in a fair manner, ensuring item exposure does not exceed a specified limit.
    
    Args:
    - recommendation_matrix (np.ndarray): Matrix of recommendation scores between users and items.
    - allocation_limit (float): Limit on the ratio of total allocations to total users.
    - top_k (int): Desired number of top recommendations per user.
    - exposure_history (np.ndarray, optional): Historical matrix of item exposures to users.
    """
    num_users, num_items = recommendation_matrix.shape
    users = list(range(num_users))
    
    # Initialize feasible sets based on exposure history
    feasible_sets = {user: set(range(num_items)) - set(np.where(exposure_history[user])[0]) for user in users} if exposure_history is not None else {user: set(range(num_items)) for user in users}
    
    # Calculate the maximum number of copies per item based on the allocation limit
    if not allocation_limit:
        allocation_limit = 1
    copies_per_item = int(allocation_limit * num_users * top_k / num_items)

    total_copies_to_allocate = copies_per_item * num_items - np.sum(exposure_history) if exposure_history is not None else copies_per_item * num_items
    if total_copies_to_allocate < num_users * top_k:
        print("The allocation limit is too low to allocate the desired number of items to each user.")
        # adjust the allocation limit
        new_allocation_limit = allocation_limit
        while total_copies_to_allocate < num_users * top_k:
            new_allocation_limit += 0.01
            copies_per_item = int(new_allocation_limit * num_users * top_k / num_items)
            total_copies_to_allocate = copies_per_item * num_items - np.sum(exposure_history) if exposure_history is not None else copies_per_item * num_items
        print("Adjusted allocation limit: {} from old {}".format(new_allocation_limit, allocation_limit))
    print("Copies per item: ", copies_per_item)
    print("Total copies to allocate: ", total_copies_to_allocate)
    
    
    # Determine the total rounds
    total_rounds = int(math.ceil((top_k * num_items) / num_users)) # check if it is correct
    
    
    # First round of greedy round-robin allocation
    allocations, updated_feasible_sets = greedy_round_robin(num_users, num_items, total_rounds, copies_per_item, recommendation_matrix, users, feasible_sets, top_k, exposure_history)
    # print(allocations, updated_feasible_sets)
    
    # Process allocations for users with less than top_k items
    for user in users:
        allocated_items = allocations[user]
        if len(allocated_items) < top_k:
            # print("Additional items needed for user", user)
            additional_items_needed = top_k - len(allocated_items)
            additional_items = sorted(list(updated_feasible_sets[user]), key=lambda item: recommendation_matrix[user, item], reverse=True)[:additional_items_needed]
            allocations[user].extend(additional_items)

    return allocations


def random_dispersal(topk_recommendation, shown_history, read_history, threshold=0.4, target_clusters='all', allocation_limit=1, top_k=5):
    """
    topk_recommendation: dict {user_id: list of top k item ids}
    allocation_limit: number of items to be swapped with items from other clusters
    """
    # spectral coclustering on exposure history
    model = ExtendedSpectralCoclustering(n_clusters=3, svd_method='arpack') # TODO seed?
    matrix = shown_history[:, :]
    try:
        model.fit(matrix)
    except:
        # add small noise
        regularization_strength = 1e-5  
        matrix += regularization_strength
        model.fit(matrix)
    sil_score, row_results_dict, col_results_dict = bicluster_silhouette_score_z(model.z_, model.row_labels_, model.column_labels_)
    print(sil_score)
    print(row_results_dict)
    print(col_results_dict)

    # monitor the homogeneity and silhouette score of the biclusters, if over threshold, moderate the recommendation matrix
    # TODO decide threshold value, absolute or relevant to the initial matrix
    # TODO decide threshold above any cluster or the overall score?
    new_recommendation = {k:v for k, v in topk_recommendation.items()}
    if sil_score > threshold:
        # do random dispersal
        # TODO decide whether to do that for all 3 user clusters or just the tightest ones
        if target_clusters == 'all':            
            new_recommendation = random_swap(topk_recommendation, read_history, model,swap_max=allocation_limit, total_rec=top_k, targets=[0,1,2])
        elif target_clusters == 'highest':
            # the row cluster with the highest score in the row_results_dict
            # target_cluster = [max(row_results_dict, key=lambda x: row_results_dict[x][0])]
            # the row cluster with scores higher than sil_score
            target_clusters = [k for k, v in row_results_dict.items() if v[0] > sil_score]
            partial_recommendation = random_swap(topk_recommendation, read_history, model, swap_max=allocation_limit, total_rec=top_k, targets=target_clusters)
            # update new_recommendation, if same key, replace with new value from partial
            new_recommendation.update(partial_recommendation)
    return new_recommendation


def random_swap(topk_recommendation, read_history, biclustering_model, swap_max, total_rec, targets):
    # if target_clusters == 'all': for each cluster, for each user, from the -1 item in recommended list, get item cluster label, if same with user cluster, randomly choose an item from another cluster and swap
    user_labels = biclustering_model.row_labels_
    item_labels = biclustering_model.column_labels_

    user_clusters = np.unique(user_labels)
    assert set(targets).issubset(user_clusters)

    num_users, num_items = read_history.shape
    users = list(range(num_users))
    feasible_sets = {user: set(range(num_items)) - set(np.where(read_history[user])[0]) for user in users}
    
    new_recommendation = {}
    
    for user_cluster in targets:
        cluster_members = np.where(user_labels == user_cluster)[0]
        # print('Cluster members:', cluster_members)
        for u in cluster_members:
            cur_rec = topk_recommendation[u]
            new_rec = list(cur_rec.copy())
            swapped = []
            while len(swapped) < swap_max and new_rec:
                i = new_rec.pop(-1)
                item_cluster = item_labels[i]
                if item_cluster == user_cluster:
                    # feasible set = items from either of the 2 other clusters and not in user's read history
                    feasible_set = feasible_sets[u] - set(np.where(item_labels == item_cluster)[0])
                    if not feasible_set:
                        print('No feasible set for user', u)
                        continue
                    # randomly choose an item from the feasible set
                    new_item = np.random.choice(list(feasible_set))
                    swapped.append(new_item)
                    feasible_sets[u].remove(new_item)
            # print(len(swapped))
            updated_rec = cur_rec if not swapped else list(cur_rec)[:-len(swapped)] + swapped
            assert len(updated_rec) == total_rec
            # print(topk_recommendation[u], updated_rec)
            new_recommendation[u] = updated_rec

    return new_recommendation


def sim_based_dispersal(topk_recommendation, shown_history, read_history, threshold=0.4, target_clusters='all', allocation_limit=1, top_k=5):
    """
    topk_recommendation: dict {user_id: list of top k item ids}
    allocation_limit: number of items to be swapped with items from other clusters
    """
    # spectral coclustering on exposure history
    model = FuzzySpectralCoclustering(n_clusters=3, svd_method='arpack') # TODO seed?
    matrix = shown_history[:, :]
    try:
        model.fit(matrix)
    except:
        # add small noise
        regularization_strength = 1e-5  
        matrix += regularization_strength
        model.fit(matrix)
    sil_score, row_results_dict, col_results_dict = bicluster_silhouette_score_z(model.z_, model.row_labels_, model.column_labels_)
    print(sil_score)
    print(row_results_dict)
    print(col_results_dict)

    # monitor the homogeneity and silhouette score of the biclusters, if over threshold, moderate the recommendation matrix
    # TODO decide threshold value, absolute or relevant to the initial matrix
    # TODO decide threshold above any cluster or the overall score?
    new_recommendation = {k:v for k, v in topk_recommendation.items()}
    if sil_score > threshold:
        # do random dispersal
        # TODO decide whether to do that for all 3 user clusters or just the tightest ones
        if target_clusters == 'all':            
            new_recommendation = sim_swap(topk_recommendation, read_history, model,swap_max=allocation_limit, total_rec=top_k, targets=[0,1,2])
        elif target_clusters == 'highest':
            # the row cluster with the highest score in the row_results_dict
            # target_cluster = [max(row_results_dict, key=lambda x: row_results_dict[x][0])]
            # TODO: the row cluster with scores higher than sil_score or row avg????
            row_mean = np.mean([v[0] for k, v in row_results_dict.items()])
            target_clusters = [k for k, v in row_results_dict.items() if v[0] > row_mean]
            partial_recommendation = sim_swap(topk_recommendation, read_history, model, swap_max=allocation_limit, total_rec=top_k, targets=target_clusters)
            # update new_recommendation, if same key, replace with new value from partial
            new_recommendation.update(partial_recommendation)
    return new_recommendation


def sim_swap(topk_recommendation, read_history, biclustering_model, swap_max, total_rec, targets):
    # if target_clusters == 'all': for each cluster, for each user, from the -1 item in recommended list, get item cluster label, if same with user cluster, randomly choose an item from another cluster and swap
    user_labels = biclustering_model.row_labels_
    item_labels = biclustering_model.column_labels_
    
    membership_matrix = biclustering_model.membership_matrix_
    n_clusters = membership_matrix.shape[1]
    n_rows = user_labels.shape[0]
    row_memberships = membership_matrix[:n_rows]  # User memberships
    col_memberships = membership_matrix[n_rows:]  # Item memberships
    # print(membership_matrix.shape)
    # print(n_clusters, n_rows)

    user_clusters = np.unique(user_labels)
    assert set(targets).issubset(user_clusters)

    num_users, num_items = read_history.shape
    users = list(range(num_users))
    feasible_sets = {user: set(range(num_items)) - set(np.where(read_history[user])[0]) for user in users}
    # print(feasible_sets.keys())
    
    new_recommendation = {}
    
    for user_cluster in targets:
        cluster_members = np.where(user_labels == user_cluster)[0]
        # print('Cluster members:', cluster_members)
        for u in cluster_members:
            cur_rec = topk_recommendation[u]
            new_rec = list(cur_rec.copy())

            swapped = []
            while len(swapped) < swap_max and new_rec:
                i = new_rec.pop(-1)
                item_cluster = item_labels[i]
                if item_cluster == user_cluster:
                    # feasible set = items from either of the 2 other clusters and not in user's read history
                    feasible_set = feasible_sets[u] - set(np.where(item_labels == item_cluster)[0])
                    if not feasible_set:
                        print('No feasible set for user', u)
                        continue
                    # Items with the highest membership to the other cluster
                    # TODO decide how to select the candidate
                    # other_cluster = [c for c in range(3) if c != user_cluster][len(swapped)%(n_clusters-1)]
                    other_cluster = np.random.choice([c for c in range(n_clusters) if c != user_cluster])
                    # print(other_cluster)
                    # print(feasible_set)
                    # Convert feasible_set to a list and sort it to maintain order
                    feasible_list = sorted(list(feasible_set))

                    # Extract memberships of the feasible items for the other cluster
                    feasible_memberships = col_memberships[feasible_list, other_cluster]

                    # Sort the feasible memberships, but keep the indices relative to the original col_memberships
                    sorted_indices = np.argsort(feasible_memberships)

                    # Get the last index from sorted_indices, which corresponds to the highest membership
                    # This index is relative to feasible_list, so we use it to get the original index
                    highest_membership_idx = feasible_list[sorted_indices[-1]]

                    # Now, highest_membership_idx corresponds to the original index in col_memberships
                    new_item = highest_membership_idx

                    # print()
                    swapped.append(new_item)
                    # print(feasible_sets.keys())
                    # print()
                    feasible_sets[u].remove(new_item)

            # # print(len(swapped))
            updated_rec = cur_rec if not swapped else list(cur_rec)[:-len(swapped)] + swapped
            assert len(updated_rec) == total_rec
            # print(topk_recommendation[u], updated_rec)
            new_recommendation[u] = updated_rec

    return new_recommendation


def moderate(moderator, recommendation_matrix, topk_recommendation, read_history, shown_history, alpha=0.8, k=5, time_limit=600, sil_threshold=0.4):
    if moderator == 'mip':
        result = popularity_suppression_linear(recommendation_matrix, exposure_history=read_history, alpha1=alpha, k=k, time_limit=time_limit, options=None)
    elif moderator == 'rr':
        result = fair_rec_allocation(recommendation_matrix, exposure_history=read_history, allocation_limit=alpha, top_k=k)
    elif moderator == 'rd':
        # TODO set threshold to be 0.8 * initial matrix's silhouette score?
        result = random_dispersal(topk_recommendation, shown_history=shown_history, read_history=read_history, threshold=sil_threshold, target_clusters='all', allocation_limit=alpha, top_k=k)
    elif moderator == 'rdtop':
        # TODO set threshold to be 0.8 * initial matrix's silhouette score?
        result = random_dispersal(topk_recommendation, shown_history=shown_history, read_history=read_history, threshold=sil_threshold, target_clusters='highest', allocation_limit=alpha, top_k=k)
    elif moderator == 'sd':
        result = sim_based_dispersal(topk_recommendation, shown_history=shown_history, read_history=read_history, threshold=sil_threshold, target_clusters='all', allocation_limit=alpha, top_k=k)
    elif moderator == 'sdtop':
        result = sim_based_dispersal(topk_recommendation, shown_history=shown_history, read_history=read_history, threshold=sil_threshold, target_clusters='highest', allocation_limit=alpha, top_k=k)
    else:
        raise ValueError('Invalid moderator')
    
    return result