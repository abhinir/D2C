"""
Suggested first settings for the Panda Box-iLQR benchmark.
"""

import numpy as np


# MuJoCo / control discretization
PHYSICS_DT = 0.002
N_SUBSTEPS = 10
CONTROL_DT = PHYSICS_DT * N_SUBSTEPS  # 0.02 s

HORIZON = 100
FINAL_TIME = HORIZON * CONTROL_DT      # 2.0 s

# Physical torque limits [Nm]
U_MAX = np.array(
    [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]
)
U_MIN = -U_MAX

# Model-free linearization
NUM_ROLLOUTS = 120
CONTROL_PERTURBATION_FRACTION = 0.01

# q [rad], qdot [rad/s]
STATE_STD = np.concatenate(
    (
        2e-4 * np.ones(7),
        2e-3 * np.ones(7),
    )
)

RIDGE = 1e-8

# Backward-pass regularization
REG_INIT = 1e-6
REG_MIN = 1e-9
REG_MAX = 1e10
REG_INCREASE = 10.0
REG_DECREASE = 0.2

# Typical line-search schedule
ALPHAS = [
    1.0,
    0.5,
    0.25,
    0.125,
    0.0625,
    0.03125,
    0.015625,
]

# Convergence diagnostic
FEEDFORWARD_TOL = 1e-4
