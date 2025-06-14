import numpy as np

def randomized_gossip_average(adjacency, values, num_iters=1000, verbose=False):
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
    for it in range(num_iters):
        # Randomly select an edge (i, j) from the adjacency matrix
        i = np.random.randint(N)
        neighbors = np.where(adjacency[i])[0]
        if len(neighbors) == 0:
            continue  # skip isolated node
        j = np.random.choice(neighbors)
        # Both nodes update to their average
        avg = 0.5 * (x[i] + x[j])
        x[i] = avg
        x[j] = avg
        history.append(x.copy())
        if verbose and it % 100 == 0:
            print(f"Iteration {it}: mean={np.mean(x):.4f}, std={np.std(x):.4f}")
    return history, real_avg