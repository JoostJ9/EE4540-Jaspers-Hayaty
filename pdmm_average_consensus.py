import numpy as np


def pdmm_average_consensus(adjacency, values, num_iters=100, c=1.0, verbose=False, Broadcast=False, transmission_loss = 0.0, synchronous = False):
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
    transmissions = [0] # List to keep track of transmissions per iteration
    
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

    tx = 0


    for k in range(num_iters):
        x_new = np.zeros_like(x)
        if synchronous:
            active_nodes = range(N)
        else:
            active_nodes = np.random.choice(range(N), int(0.5*N))  # Randomly select 50% of nodes for update

        # x update
        for i in active_nodes:
            neighbours = np.where(adjacency[i])[0]
            z_sum = 0.0
            
            for j in neighbours:
                A_ij = 1.0 if i < j else -1.0
                z_sum += A_ij * z[(i, j)]
            x_new[i] = (a[i] - z_sum) / (1 + c * degrees[i])

        # y update
        if Broadcast:   #with broadcasting x is sent
            for j in active_nodes: 
                neighbours = np.where(adjacency[j])[0]
                tx += 1
                if transmission_loss > 0:
                        loss = np.random.choice([0,1], p=[1-transmission_loss, transmission_loss])
                        if loss == 1:
                            continue
                for i in neighbours:  # i is the sender
                    A_ij = 1.0 if i < j else -1.0
                    y[(i, j)] = z[(i, j)] + 2 * c * A_ij * x_new[i]

        else:       # with unicast first y is computed then they are sent. 
            for i in active_nodes:
                neighbours = np.where(adjacency[i])[0]
                for j in neighbours:
                    if transmission_loss > 0:
                        loss = np.random.choice([0,1], p=[1-transmission_loss, transmission_loss])
                        if loss == 1:
                            continue
                    A_ij = 1.0 if i < j else -1.0
                    y[(i, j)] = z[(i, j)] + 2 * c * A_ij * x_new[i]
                    tx += 1  
        # z update
        for i in active_nodes:
            neighbours = np.where(adjacency[i])[0]
            for j in neighbours:
                z[(j, i)] = y[(i, j)]

        x = x_new.copy()
        history.append(x.copy())
        transmissions.append(tx)

        err = np.linalg.norm(x - real_avg)/np.linalg.norm(real_avg)  # Calculate the normalized error from the real average
        if err < 1e-12:
            if verbose:
                print(f"Converged at iteration {k} with max error {err:.14f}")
            break
        if verbose and k % 100 == 0:
            print(f"Iter {k}: max error = {err:.10f}")

    return k, real_avg, history, transmissions