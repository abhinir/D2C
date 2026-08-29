import numpy as np

state_dimension = 2
control_dimension = 1
obs_dimension = state_dimension

# Cost parameters for nominal design
Q = 9*np.array([[1,0],[0,1]])
Q_terminal = 9000*np.array([[1,0],[0,1]])
Q_final = Q_terminal
R = .005*np.eye(control_dimension)

# Number of substeps in simulation
ctrl_state_freq_ratio = 1
dt = 0.1
horizon = 20 
nominal_init_stddev = 0.1

alpha = 1

# Cost parameters for feedback design
W_x_LQR = 10*np.eye(state_dimension*state_dimension)
W_u_LQR = 2*np.eye(2*control_dimension*control_dimension)
W_x_LQR_f = 100*np.eye(state_dimension*state_dimension)

# D2C parameters
feedback_n_samples = 50
n_substeps = 1