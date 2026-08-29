import os
import numpy as np
import csv

def save_to_csv(x, u, cost, folder_name="data"):
    """
    Save x, u, cost arrays to CSV files in a folder.

    x:    numpy array of shape (n, N+1) -> states over horizon
    u:    numpy array of shape (m, N)   -> controls over horizon
    cost: numpy array of shape (K,)    -> cost over iterations
    """

    # Ensure cost is 1D
    cost = np.asarray(cost).reshape(-1)

    # Create folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # File paths
    x_path    = os.path.join(folder_name, "states_x.csv")
    u_path    = os.path.join(folder_name, "controls_u.csv")
    cost_path = os.path.join(folder_name, "cost.csv")

    # Save x (states)
    # Rows: time steps (0..N), Columns: state-1, state-2, ..., state-n
    x_T = x.T  # shape (N+1, n)
    with open(x_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        for t, row in enumerate(x_T):
            writer.writerow([t] + list(row))
        # Header
        n = x.shape[0]
#        header = ["t"] + [f"state-{i+1}" for i in range(n)]
#        writer.writerow(header)
        # Data
        

    # Save u (controls)
    # Rows: time steps (0..N-1), Columns: control-1, ..., control-m
    u_T = u.T  # shape (N, m)
    with open(u_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        for t, row in enumerate(u_T):
            writer.writerow([t] + list(row))
#        m = u.shape[0]
#        header = ["t"] + [f"control-{j+1}" for j in range(m)]
#        writer.writerow(header)
        

    # Save cost
    # Rows: iterations, Columns: cost
    with open(cost_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        for k, c in enumerate(cost):
            writer.writerow([k, c])
#
#        writer.writerow(["iteration", "cost"])
        

