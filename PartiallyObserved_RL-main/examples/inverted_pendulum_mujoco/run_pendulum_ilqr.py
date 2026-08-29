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

from sim_pendulum import SimulatePendulum
from pendulum_params import *
from main_ilqr import iLQR
from ltv_sys_id import LTV_SysID

class RunPendulum(SimulatePendulum):

    def __init__(self, state_dimension, control_dimension, dt, model_path=None):
        SimulatePendulum.__init__(self, state_dimension, control_dimension, dt, model_path=str(model_path))

    def simulate_step(self,x,u):
        return self.simulate_trajectory(x, u)[-1]


if __name__=="__main__":

    cwd = os.getcwd()
    path_to_pendulum = Path(cwd)/"examples/inverted_pendulum_mujoco"
    MODEL = path_to_pendulum/"models/pendulum.xml"

    path_to_export = path_to_pendulum/"Inverted_Pendulum_Experiments/exp_1"
    path_to_policy_file = path_to_export/"pendulum_policy.txt"
    path_to_cost_file = path_to_export / "training_cost_data.txt"
    path_to_training_cost_fig = path_to_export/"episodic_cost_training.png"
    path_to_traj_fig = path_to_export/"optimal_traj.png"
    
    init_state = np.zeros((state_dimension,1))
    init_state[0] = np.pi
    final_state = np.zeros((state_dimension, 1))

    print('Initial phase : \n', init_state)
    print('Goal phase : \n', final_state)
	
    # No. of ILQR iterations to run
    n_iterations = 50

    # Create model instance
    run_pendulum = RunPendulum(state_dimension, control_dimension, dt, MODEL)

    # Create iLQR instance
    ilqr = iLQR(run_pendulum, state_dimension, control_dimension, alpha, horizon, init_state, final_state, Q, Q_final, R, 
                nominal_init_stddev, n_sys_id_samples=50, pert_sys_id_sigma=1e-1, arma_sys_id_flag = False)
    ilqr.iterate_ilqr(n_iterations)


    ilqr.plot_episodic_cost_history(path_to_training_cost_fig)
    ilqr.save_policy(path_to_policy_file)
    ilqr.save_cost(path_to_cost_file)

    # Check and Simulate the obtained policy
    run_pendulum.simulate_trajectory(y_init = init_state.flatten(), u = ilqr.U.flatten(), horizon=horizon)
    run_pendulum.draw_figure(path_to_traj_fig)
