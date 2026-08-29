import numpy as np

state_dimension = 37
control_dimension = 12
obs_dimension = state_dimension

# Cost parameters for nominal design
Q = np.diag([
    # Base position (xyz)
    10, 10, 2000,           # Emphasize height maintenance (z)
    
    # Base orientation (quaternion w,x,y,z)
    100, 100, 100, 100,    # Keep upright, penalize tilting
    
    # Joint positions (12 joints: FL, FR, RL, RR)
    10, 10, 10,            # FL: hip, thigh, calf
    10, 10, 10,            # FR
    10, 10, 10,            # RL
    10, 10, 10,            # RR
    
    # Base velocities (vx, vy, vz)
    1, 50, 10,             

    # Base angular velocities (wx, wy, wz)
    50, 50, 50,            
    
    # Joint velocities (12)
    1, 1, 1,               
    1, 1, 1,
    1, 1, 1,
    1, 1, 1
])

Q_final = 10000 * Q
Q_terminal = Q_final
R = 1*np.eye(control_dimension)

# Number of substeps in simulation
ctrl_state_freq_ratio = 1
dt = 0.01
horizon = 50
nominal_init_stddev = 0.1

alpha = 1

# Cost parameters for feedback design
W_x_LQR = 10*np.eye(state_dimension*state_dimension)
W_u_LQR = 2*np.eye(2*control_dimension*control_dimension)
W_x_LQR_f = 100*np.eye(state_dimension*state_dimension)

# D2C parameters
feedback_n_samples = 2000
n_substeps = 1