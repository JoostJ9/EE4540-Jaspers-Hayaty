import numpy as np
import cvxpy as cp

def randomized_gossip_average(adjacency, values, P, num_iters=1000, verbose=False, transmissions_loss = 0, threshold=1e-12):
    """
    Runs the randomized gossip algorithm to compute the average of the initial values.
    
    Parameters:
        adjacency: (N, N) boolean array, adjacency matrix of the network
        values: (N,) array, initial values at each node
        num_iters: int, number of gossip iterations
        verbose: bool, if True, prints progress
    
    Returns:
        history: list of arrays, value at each node after each iteration
    """
    real_avg = np.mean(values)  # Calculate the true average of the initial values
    N = len(values)
    x = values.copy()
    history = [x.copy()]
    transmissions = [0]  # List to keep track of transmissions per iteration
    tx = 0  # Initialize transmission count
    for k in range(num_iters):
        # Randomly select an edge (i, j) from the adjacency matrix
        i = np.random.randint(N)
        neighbors = np.where(adjacency[i])[0]
        if len(neighbors) == 0:
            continue  # skip isolated node
        j = np.random.choice(len(P[i]), p=P[i]/np.sum(P[i]))  # Choose neighbor j based on probabilities in P
        # Both nodes update to their average
        avg = 0.5 * (x[i] + x[j])
        if transmissions_loss > 0:
            loss1 = np.random.choice([0, 1], p=[1-transmissions_loss, transmissions_loss])
            loss2 = np.random.choice([0, 1], p=[1-transmissions_loss, transmissions_loss])
        else:
            loss1 = 0
            loss2 = 0
        if loss1 == 0:
            x[i] = avg
        if loss2 == 0:
            x[j] = avg
        tx += 2  # Count the number of transmissions
        history.append(x.copy())
        transmissions.append(tx)
        err = np.linalg.norm(x - real_avg)/np.linalg.norm(real_avg)  # Calculate the normalized error from the real average
        if err < threshold:
            if verbose:
                print(f"Converged at iteration {k} with max error {err:.14f}")
            break
        if verbose and k % 100 == 0:
            print(f"Iter {k}: max error = {err:.10f}")
    return k, history, real_avg, transmissions

def compute_P_matrix(adjacency):
    """
    Computes the P matrix for the randomized gossip algorithm based on the adjacency matrix.
    
    Parameters:
    - adjacency: (N, N) boolean array, adjacency matrix of the network
    
    Returns:
    - P: (N, N) matrix where P[i, j] is the probability of node i communicating with node j
    """
    N = adjacency.shape[0]
    Wij_list = []
    for i in range(N):
        for j in range(N):
            if adjacency[i, j]:
                e_i = np.zeros((N, 1)); e_i[i] = 1
                e_j = np.zeros((N, 1)); e_j[j] = 1
                W_ij = np.eye(N) - 0.5 * (e_i - e_j) @ (e_i - e_j).T
                Wij_list.append(W_ij)
    
    Wij_flat = np.array([W.flatten() for W in Wij_list])
    p = cp.Variable(len(Wij_list), nonneg=True)

    W_bar_vec = p @ Wij_flat / N
    W_bar = cp.reshape(W_bar_vec, (N, N), order='F')

    # Constraints: each row sum of P (from p) must be 1
    constraints = []
    cnt = 0
    for i in range(N):
        row_expr = 0
        for j in range(N):
            if adjacency[i, j]:
                row_expr += p[cnt]
                cnt += 1
        constraints.append(row_expr == 1)

    one_vec = np.ones((N, 1))
    W_proj = W_bar - (one_vec @ one_vec.T) / N

    objective = cp.Minimize(cp.lambda_max(W_proj))
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS)

    # Reconstruct P matrix from p
    P = np.zeros((N, N))
    cnt = 0
    for i in range(N):
        for j in range(N):
            if adjacency[i, j]:
                P[i, j] = p[cnt].value
                cnt += 1

    return np.round(P, 4)

def calculate_W_bar(P, adjacency):
    """
    Calculate the W_bar matrix based on the P matrix and adjacency matrix.
    
    Parameters:
        P: (N, N) matrix, probability matrix for gossip communication
        adjacency: (N, N) boolean array, adjacency matrix of the network
    
    Returns:
        W_bar: (N, N) matrix, average gossip weight matrix
    """
    N = adjacency.shape[0]
    W_bar = np.zeros_like(adjacency, dtype='float64')

    for i in range(N):
        for j in range(N):
            if adjacency[i, j]:
                e_i = np.zeros((N, 1)); e_i[i] = 1
                e_j = np.zeros((N, 1)); e_j[j] = 1
                W_ij = np.eye(N) - 0.5 * (e_i - e_j) @ (e_i - e_j).T
                W_bar += P[i, j] * W_ij
    
    return W_bar / N