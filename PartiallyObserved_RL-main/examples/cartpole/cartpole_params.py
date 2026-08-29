import numpy as np

state_dimension = 4
control_dimension = 1
obs_dimension = state_dimension

# Cost parameters for nominal design
# Q = np.eye(state_dimension)#np.diag([10.0, 1.0, 40.0, 1.5])
Q = 10*np.eye(state_dimension)#np.diag([10.0, 40.0, 1.0, 1.5])       
R = np.diag([0.005])                       
Q_terminal = 1000*np.eye(state_dimension)#np.diag([2700,9000,2700,2700])#1000*Q#900.0*np.diag([3.0, 3.0, 10.0, 3.0])
Q_final = Q_terminal

# Number of substeps in simulation
ctrl_state_freq_ratio = 1
dt = 0.1
horizon = 30 #800
nominal_init_stddev = 0.1

q = 1
q_u = 1
alpha = 1.0

# Cost parameters for feedback design
W_x_LQR = 10*np.eye(state_dimension*state_dimension)
W_u_LQR = 2*np.eye(2*control_dimension*control_dimension)
W_x_LQR_f = 100*np.eye(state_dimension*state_dimension)

# D2C parameters
feedback_n_samples = 20
n_substeps = 1