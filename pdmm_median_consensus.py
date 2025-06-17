import numpy as np
import matplotlib.pyplot as plt

def median_consensus(adjacency, values, num_iters=100, c=1.0, verbose=False, Broadcast = True, transmission_loss = 0, min_error = 1e-2, synchronous = True):
    """
    Median consensus via PDMM 

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
    x = values.copy()
    history = [s.copy()]
    degrees = np.sum(adjacency, axis=1)
    real_median = np.median(s) # Calculate the true median of the initial values
    transmissions = [0]
    

    if np.any(degrees == 0):
        raise ValueError("Some nodes have no neighbors; graph must be connected.")

    z = {}
    for i in range(N):
        for j in range(N):
            if adjacency[i, j] > 0:
                z[(i, j)] = np.random.normal(s[i], 1)

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

            if (-1 - zsum)/(c*len(neighbors)) > s[i]:
                x_new[i] = (-1 - zsum)/(c*len(neighbors))
            elif (1 - zsum)/(c*len(neighbors)) < s[i]:
                x_new[i] = (1 - zsum)/(c*len(neighbors))
            else: 
                x_new[i] = s[i]
        if synchronous:
            active_nodes = range(N)
        else:
            active_nodes = np.random.choice(range(N), int(0.5*N))  # Randomly select 50% of nodes for update

        # z update
        z_prev = z.copy()
        for i in active_nodes:
            # With broadcast, transmission loss affects transmission of x_i to all neighbors
            if (Broadcast and transmission_loss > 0):
                    lost = np.random.choice([0,1], p=[1-transmission_loss, transmission_loss])
                    if (lost == 1):
                        tx += 1
                        continue        #break out of the loop if transmission is lost 

            for j in np.where(adjacency[i])[0]:
                # With unicast, transmission loss only affects the transmission to one neighbor              
                if Broadcast:
                    A = 1 if i < j else -1
                    z[(j, i)] = 0.5*z_prev[(j, i)] + 0.5*(z_prev[(i, j)] + 2*c*A*x_new[i])
                else:
                    if (transmission_loss > 0):
                        lost = np.random.choice([0,1], p=[1-transmission_loss, transmission_loss])
                        if (lost == 1):
                            tx += 1
                            continue    #break out of secundar loop if transmission is lost
                    A = 1 if i < j else -1
                    z[(j, i)] = 0.5*z_prev[(j, i)] + 0.5*(z_prev[(i, j)] + 2*c*A*x_new[i])
                    tx += 1

            if (Broadcast == True):
                tx += 1


        x = x_new.copy()
        history.append(x.copy())
        transmissions.append(tx)

        err = np.linalg.norm(x - real_median)/np.linalg.norm(real_median)

        if verbose and k % 100 == 0:
            print(f"Iter {k}: max deviation from median = {err:.6f}")

        if err < min_error:
            if verbose:
                print(f"Converged at iteration {k} with max error {err:.14f}")
            break

    return k + 1, real_median, np.array(history), transmissions