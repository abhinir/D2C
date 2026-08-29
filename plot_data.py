import numpy as np
import matplotlib.pyplot as plt

def plot_results(x, u, cost):
    """
    x:    numpy array of shape (n, N+1)   -> states over horizon
    u:    numpy array of shape (m, N)     -> controls over horizon
    cost: numpy array of shape (K,) or (K,1) -> cost over iterations
    """

    # State and control dimensions
    n, Np1 = x.shape      # Np1 = N + 1 (time steps for states)
    m, N   = u.shape      # N (time steps for controls)

    # Time indices for plotting
    t_x = np.arange(Np1)  # 0, 1, ..., N for states
    t_u = np.arange(N)    # 0, 1, ..., N-1 for controls
    it  = np.arange(cost.shape[0])  # iterations for cost

    # Plot states
    plt.figure(figsize=(8, 4))
    for i in range(n):
        plt.plot(t_x, x[i, :], label=f"state-{i+1}")
    plt.xlabel("Horizon (time step)")
    plt.ylabel("States")
    plt.title("States over horizon")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Plot controls
    plt.figure(figsize=(8, 4))
    for j in range(m):
        plt.plot(t_u, u[j, :], label=f"control-{j+1}")
    plt.xlabel("Horizon (time step)")
    plt.ylabel("Controls")
    plt.title("Controls over horizon")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Plot cost
    plt.figure(figsize=(8, 4))
    plt.plot(it, cost, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Cost per iteration")
    plt.grid(True)
    plt.tight_layout()

    plt.show()


# Example usage (remove or replace with your data):
