import numpy as np

def median_consensus(adjacency, values, num_iters=100, c=1.0, verbose=False,
                     Broadcast=True, transmission_loss=0, min_error=1e-12, synchronous=True):
    """
    Median consensus via PDMM with advanced error metric (projected + variance-based).
    """

    N = len(values)
    s = values.copy()
    x = values.copy()
    history = [x.copy()]
    degrees = np.sum(adjacency, axis=1)
    transmissions = [0]
    errors = []

    # Median interval boundaries (for even n)
    s_sorted = np.sort(s)
    if N % 2 == 0:
        x_l = s_sorted[N // 2 - 1]
        x_u = s_sorted[N // 2]
    else:
        x_l = x_u = s_sorted[N // 2]

    if np.any(degrees == 0):
        raise ValueError("Some nodes have no neighbors; graph must be connected.")

    # Initialize z variables
    z = {}
    for i in range(N):
        for j in range(N):
            if adjacency[i, j] > 0:
                z[(i, j)] = np.random.normal(0, 1)  # zero-mean noise is better

    tx = 0

    for k in range(num_iters):
        x_new = np.zeros_like(x)

        # x update
        for i in range(N):
            neighbors = np.where(adjacency[i])[0]
            zsum = 0
            for j in neighbors:
                A = 1 if i < j else -1
                zsum += A * z[(i, j)]

            proj = (-1 - zsum) / (c * len(neighbors))
            upper = (1 - zsum) / (c * len(neighbors))

            if proj > s[i]:
                x_new[i] = proj
            elif upper < s[i]:
                x_new[i] = upper
            else:
                x_new[i] = s[i]

        if synchronous:
            active_nodes = range(N)
        else:
            active_nodes = np.random.choice(N, int(0.5 * N), replace=False)

        # z update
        z_prev = z.copy()
        for i in active_nodes:
            if Broadcast and transmission_loss > 0:
                if np.random.rand() < transmission_loss:
                    tx += 1
                    continue

            for j in np.where(adjacency[i])[0]:
                if not Broadcast and transmission_loss > 0:
                    if np.random.rand() < transmission_loss:
                        tx += 1
                        continue

                A = 1 if i < j else -1
                z[(j, i)] = 0.5 * z_prev[(j, i)] + 0.5 * (z_prev[(i, j)] + 2 * c * A * x_new[i])
                if not Broadcast:
                    tx += 1
            if Broadcast:
                tx += 1

        x = x_new.copy()
        history.append(x.copy())
        transmissions.append(tx)

        epsilon_vec = np.zeros(N)
        for i in range(N):
            if x[i] < x_l:
                epsilon_vec[i] = x_l - x[i]
            elif x[i] > x_u:
                epsilon_vec[i] = x[i] - x_u
            else:
                epsilon_vec[i] = 0

        projected_error = np.linalg.norm(epsilon_vec)
        error = (projected_error + np.var(x)) / (0.5 * (x_l + x_u))
        errors.append(error)

        if verbose and k % 100 == 0:
            print(f"Iter {k}: projected+var error = {error:.6e}")

        if error < min_error:
            if verbose:
                print(f"Converged at iteration {k} with custom error {error:.2e}")
            break

    return k + 1, [x_l,x_u], np.array(history), transmissions, errors




