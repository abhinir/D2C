"""
After your iLQR / Box-iLQR solver returns x_trajectory and u_trajectory,
save them in the format expected by render_panda.py.
"""

import numpy as np


def save_panda_solution(
    x_trajectory,
    u_trajectory,
    filename="panda_solution.npz",
):
    np.savez(
        filename,
        x_trajectory=x_trajectory,
        u_trajectory=u_trajectory,
    )

    print(f"Saved {filename}")
