import numpy as np

state_dimension = 27
control_dimension = 6
obs_dimension = state_dimension

# Cost parameters for nominal design
Q = 9*np.diag(np.concatenate([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], np.zeros((13,))]))
Q_terminal = 1500*np.diag(np.concatenate([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], np.zeros((13,))]))
Q_final = Q_terminal
R = 0.1*np.eye(control_dimension)

# Cost parameters for ARMA design
q = 1
q_u = 1
C = np.eye(state_dimension)
aug_dim = obs_dimension * q + state_dimension * (q_u - 1)
Q_aug = np.zeros((aug_dim, aug_dim))
Q_terminal_aug = np.zeros((aug_dim, aug_dim))
Q_aug[:4, :4] = np.diag([20, 20, 20, 1])
Q_terminal_aug[:4, :4] = 1000*np.diag([20, 20, 20, 3]) 
Q_final_aug = Q_terminal_aug


# Number of substeps in simulation
ctrl_state_freq_ratio = 1
dt = 0.005
horizon = 600 
nominal_init_stddev = 0.1

alpha = 1

# Cost parameters for feedback design
W_x_LQR = 10*np.eye(state_dimension*state_dimension)
W_u_LQR = 2*np.eye(2*control_dimension*control_dimension)
W_x_LQR_f = 100*np.eye(state_dimension*state_dimension)

# D2C parameters
feedback_n_samples = 50
n_substeps = 5