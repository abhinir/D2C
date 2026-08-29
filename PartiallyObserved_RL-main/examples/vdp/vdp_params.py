import numpy as np

mu = 0.2

state_dimension = 2
control_dimension = 1
obs_dimension = state_dimension

# Cost parameters for nominal design
Q = 9*np.eye(state_dimension)
Q_terminal = 9000*np.eye(state_dimension)
Q_final = Q_terminal
R = .05*np.eye(control_dimension)

# Number of substeps in simulation
ctrl_state_freq_ratio = 1
dt = 0.1
horizon = 20 #800
nominal_init_stddev = 0.1

q = 2
q_u = 2
alpha = 0.7

# C = np.eye(2)
C = np.array([1.0, 0]).reshape(1,2)
n_z = C.shape[0]

n_aug = n_z*q+control_dimension*(q_u-1)
Q_aug = np.zeros((n_aug, n_aug))
# Q_aug[0:n_z,0:n_z] = Q[0:n_z,0:n_z]
qv = 9.0
Q_aug[0:2,0:2] = np.array([[9+qv,-qv],[-qv,qv]])
Q_final_aug = 1000*Q_aug

# Cost parameters for feedback design
W_x_LQR = 10*np.eye(state_dimension*state_dimension)
W_u_LQR = 2*np.eye(2*control_dimension*control_dimension)
W_x_LQR_f = 100*np.eye(state_dimension*state_dimension)

# D2C parameters
feedback_n_samples = 20
n_substeps = 1