import numpy as np

def median_consensus(adjacency, values, num_iters=100, c=1.0, verbose=False):
    """
    Median consensus via PDMM using L1-norm minimization.

    Parameters:
        adjacency: (N, N) boolean adjacency matrix
        values: (N,) array of private scalar values (si)
        num_iters: number of iterations
        c: convergence parameter
        verbose: whether to print diagnostics

    Returns:
        history: list of node values over time
    """
    N = len(values)
    s = values.copy()
    x = np.zeros_like(s)
    history = [x.copy()]
    degrees = np.sum(adjacency, axis=1)

    if np.any(degrees == 0):
        raise ValueError("Some nodes have no neighbors; graph must be connected.")

    # A_ij = +1 if i < j, -1 if i > j
    def A(i, j): return 1.0 if i < j else -1.0

    z = {}
    for i in range(N):
        for j in np.where(adjacency[i])[0]:
            if (i, j) not in z:
                z[(i, j)] = np.random.normal(loc=A(i, j), scale=0.1)
                z[(j, i)] = -z[(i, j)]

    for t in range(num_iters):
        x_new = np.zeros_like(x)

        # x update
        for i in range(N):
            neighbors = np.where(adjacency[i])[0]
            zsum = sum(A(i, j) * z[(i, j)] for j in neighbors)
            candidate = -1 - zsum / (c * degrees[i])
            if candidate > s[i]:
                x_new[i] = candidate
            else:
                candidate = 1 - zsum / (c * degrees[i])
                if candidate < s[i]:
                    x_new[i] = candidate
                else:
                    x_new[i] = s[i]

        # z update
        for i in range(N):
            for j in np.where(adjacency[i])[0]:
                z[(j, i)] = 0.5 * z[(j, i)] + 0.5 * (z[(i, j)] + 2 * c * A(i, j) * x_new[i])

        x = x_new.copy()
        history.append(x.copy())

        if verbose and t % 10 == 0:
            med_error = np.max(np.abs(x - np.median(s)))
            print(f"Iter {t}: max deviation from median = {med_error:.6f}")

    return x, np.median(s), history
