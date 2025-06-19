import numpy as np

def pdmm_average_consensus(adjacency, values, num_iters=100, c=1.0, verbose=False,
                            Broadcast=True, transmission_loss=0.0, synchronous=True, threshold=1e-12):
    """
    PDMM for average consensus with support for synchronous/asynchronous updates,
    unicast/broadcast communication, and simulated transmission loss.

    Parameters:
        adjacency: (N, N) boolean adjacency matrix
        values: (N,) array of initial node values (a_i)
        num_iters: number of PDMM iterations
        c: step size
        verbose: whether to print convergence diagnostics
        Broadcast: if True, use broadcast instead of unicast
        transmission_loss: probability [0,1] of simulating a lost transmission
        synchronous: if True, update all nodes each iteration

    Returns:
        k: final iteration count
        real_avg: true average of the initial values
        history: list of node values at each iteration
        transmissions: list of cumulative transmission counts
    """
    real_avg = np.mean(values)
    N = len(values)
    a = values.copy()
    x = values.copy()
    history = [x.copy()]
    transmissions = [0]
    
    degrees = np.sum(adjacency, axis=1)
    if np.any(degrees == 0):
        raise ValueError("Some nodes have no neighbors, PDMM requires a connected graph.")

    y = {}
    z = {}
    for i in range(N):
        for j in np.where(adjacency[i])[0]:
            z[(i, j)] = 0.0
            z[(j, i)] = 0.0
            y[(i, j)] = 0.0
            y[(j, i)] = 0.0

    tx = 0

    for k in range(num_iters):
        x_new = x.copy()
        loss_occurred = np.zeros((N, N), dtype=bool)

        if synchronous:
            active_nodes = range(N)
        else:
            active_nodes = np.random.choice(N, size=int(0.5 * N), replace=False)

        # x update
        for i in active_nodes:
            neighbours = np.where(adjacency[i])[0]
            z_sum = sum((1.0 if i < j else -1.0) * z[(i, j)] for j in neighbours)
            x_new[i] = (a[i] - z_sum) / (1 + c * degrees[i])

        # y update
        if Broadcast:
            for i in active_nodes:  # sender
                neighbours = np.where(adjacency[i])[0]
                tx += 1
                if transmission_loss > 0:
                    loss = np.random.choice([0, 1], p=[1 - transmission_loss, transmission_loss])
                    if loss == 1:
                        loss_occurred[i, j] = True
                        continue
                for j in neighbours:
                    A_ij = 1.0 if i < j else -1.0
                    y[(i, j)] = z[(i, j)] + 2 * c * A_ij * x_new[i]
        else:
            for i in active_nodes:
                neighbours = np.where(adjacency[i])[0]
                for j in neighbours:
                    tx += 1
                    if transmission_loss > 0:
                        loss = np.random.choice([0, 1], p=[1 - transmission_loss, transmission_loss])
                        if loss == 1:
                            loss_occurred[i, j] = True
                            continue
                    A_ij = 1.0 if i < j else -1.0
                    y[(i, j)] = z[(i, j)] + 2 * c * A_ij * x_new[i]

        # z update
        for i in active_nodes:
            neighbours = np.where(adjacency[i])[0]
            for j in neighbours:
                if not loss_occurred[i, j]:
                    z[(j, i)] = y[(i, j)]

        x = x_new.copy()
        history.append(x.copy())
        transmissions.append(tx)

        err = np.linalg.norm(x - real_avg) / np.linalg.norm(real_avg)
        if err < threshold:
            if verbose:
                print(f"Converged at iteration {k} with max error {err:.2e}")
            break
        if verbose and k % 100 == 0:
            print(f"Iter {k}: max error = {err:.2e}")

    return k, real_avg, history, transmissions
