import numpy as np
import math
import sys
import os
from pathlib import Path

import sys
import os


# import Iterative_LQR
# from Iterative_LQR.src.main_ilqr import iLQR

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from sim_cartpole import SimulateCartPole
from cartpole_params import *
from main_ilqr import iLQR
from ltv_sys_id import LTV_SysID

class RunCartPole(SimulateCartPole):

    def __init__(self, state_dimension, control_dimension, dt):
        SimulateCartPole.__init__(self, state_dimension, control_dimension, dt)

    def simulate_step(self,x,u):
        return self.simulate_trajectory(x, u)[-1]


if __name__=="__main__":

    cwd = os.getcwd()
    path_to_vdp = Path(cwd)/"examples/cartpole"
    MODEL = path_to_vdp/"models/None.xml"

    path_to_export = path_to_vdp/"Cartpole_Experiments/exp_1"
    path_to_policy_file = path_to_export/"cartpole_policy.txt"
    path_to_cost_file = path_to_export / "training_cost_data.txt"
    path_to_training_cost_fig = path_to_export/"episodic_cost_training.png"
    path_to_traj_fig = path_to_export/"optimal_traj.png"
    # path_to_data = path_to_export / "vdp_D2C_data.txt"

    init_state = np.zeros((state_dimension,1))
    init_state[2] = np.pi
    final_state = np.zeros((state_dimension, 1))

    print('Initial phase : \n', init_state)
    print('Goal phase : \n', final_state)
	
    # No. of ILQR iterations to run
    n_iterations = 200

    # Create model instance
    run_cartpole = RunCartPole(state_dimension, control_dimension, dt)

    # Create iLQR instance
    ilqr = iLQR(run_cartpole, state_dimension, control_dimension, alpha, horizon, init_state, final_state, Q, Q_final, R, 
                nominal_init_stddev, n_sys_id_samples=40, pert_sys_id_sigma=1e-4, arma_sys_id_flag = False)
    ilqr.iterate_ilqr(n_iterations, u_init = None)

    print(ilqr.episodic_cost_history)

    ilqr.plot_episodic_cost_history(path_to_training_cost_fig)
    ilqr.save_policy(path_to_policy_file)
    ilqr.save_cost(path_to_cost_file)
    
    # Check and Simulate the obtained policy
    # print
    run_cartpole.simulate_trajectory(state_init = init_state.flatten(), u = ilqr.U.flatten(), horizon=horizon)
    run_cartpole.draw_figure(path_to_traj_fig)