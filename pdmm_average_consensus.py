import numpy as np


def pdmm_average_consensus(adjacency, values, num_iters=100, c=1.0, verbose=False, Broadcast=False):
    """
    Parameters:
        adjacency: (N, N) boolean adjacency matrix
        values: (N,) array of initial node values (a_i)
        num_iters: number of PDMM iterations
        c: step size (corresponds to 'c' in the formula)
        verbose: whether to print diagnostics

    Returns:
        history: list of node values over time
    """
    real_avg = np.mean(values)  # Calculate the true average of the initial values
    N = len(values)
    a = values.copy()
    x = values.copy()
    history = [x.copy()]
    
    degrees = np.sum(adjacency, axis=1) #diegrees[i] = number of neighbors of node i
    if np.any(degrees == 0):
        raise ValueError("Some nodes have no neighbors, PDMM requires a connected graph.")

    y = {}
    z = {}
    for i in range(N):
        for j in np.where(adjacency[i])[0]:
            z[(i, j)] = 0.0
            z[(j, i)] = 0.0  # Ensure symmetric access is always valid
            y[(i, j)] = 0.0
            y[(j, i)] = 0.0  # for safety in broadcast mode

    transmissions = 0


    for k in range(num_iters):
        x_new = np.zeros_like(x)

        # x update
        for i in range(N):
            neighbours = np.where(adjacency[i])[0]
            z_sum = 0.0
            
            for j in neighbours:
                A_ij = 1.0 if i < j else -1.0
                z_sum += A_ij * z[(i, j)]
            x_new[i] = (a[i] - z_sum) / (1 + c * degrees[i])

        # y update
        if Broadcast:
            for j in range(N): 
                neighbours = np.where(adjacency[j])[0]
                transmissions += 1
                for i in neighbours:  # i is the sender
                    A_ij = 1.0 if i < j else -1.0
                    b_ij = 0
                    y[(i, j)] = z[(i, j)] + 2 * c * A_ij * x_new[i] - c * b_ij

        else:
            for i in range(N):
                neighbours = np.where(adjacency[i])[0]

                for j in neighbours:
                    A_ij = 1.0 if i < j else -1.0
                    y[(i, j)] = z[(i, j)] + 2 * c * A_ij * x_new[i]
                    transmissions += 1  
        # z update
        for i in range(N):
            neighbours = np.where(adjacency[i])[0]
            for j in neighbours:
                z[(j, i)] = y[(i, j)]

        x = x_new.copy()
        history.append(x.copy())

        err = np.max(np.abs(x - real_avg))
        if err < 1e-12:
            if verbose:
                print(f"Converged at iteration {k} with max error {err:.14f}")
            break
        if verbose and k % 100 == 0:
            print(f"Iter {k}: max error = {err:.10f}")

    return k, real_avg, history, transmissions