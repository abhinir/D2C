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

from sim_fish import SimulateFish
from fish_params import *
from main_ilqr import iLQR
from ltv_sys_id import LTV_SysID

class RunFish(SimulateFish):

    def __init__(self, state_dimension, control_dimension, dt, model_path=None, n_substeps=5):
        SimulateFish.__init__(self, state_dimension, control_dimension, dt, model_path=str(model_path))

    def simulate_step(self,x,u):
        return self.simulate_trajectory(x, u)[-1]


if __name__=="__main__":

    cwd = os.getcwd()
    path_to_pendulum = Path(cwd)/"examples/fish_mujoco"
    MODEL = path_to_pendulum/"models/fish.xml"

    path_to_export = path_to_pendulum/"Fish_Experiments/exp_1"
    path_to_policy_file = path_to_export/"fish_policy.txt"
    path_to_cost_file = path_to_export / "training_cost_data.txt"
    path_to_training_cost_fig = path_to_export/"episodic_cost_training.png"
    path_to_traj_fig = path_to_export/"optimal_traj.png"
    
    init_state = np.zeros((state_dimension,1))
    init_state[3] = 1.0
    final_state = np.zeros((state_dimension, 1))
    final_state[1] = 0.4
    final_state[2] = 0.2
    final_state[3] = 1.0

    print('Initial phase : \n', init_state)
    print('Goal phase : \n', final_state)
	
    # No. of ILQR iterations to run
    n_iterations = 50

    # Create model instance
    run_fish = RunFish(state_dimension, control_dimension, dt, MODEL)

    # Create iLQR instance
    ilqr = iLQR(run_fish, state_dimension, control_dimension, alpha, horizon, init_state, final_state, Q, Q_final, R, 
                nominal_init_stddev, n_sys_id_samples=50, pert_sys_id_sigma=1e-7, arma_sys_id_flag = False)
    ilqr.iterate_ilqr(n_iterations)


    ilqr.plot_episodic_cost_history(path_to_training_cost_fig)
    ilqr.save_policy(path_to_policy_file)
    ilqr.save_cost(path_to_cost_file)

    # Check and Simulate the obtained policy
    run_fish.simulate_trajectory(y_init = init_state.flatten(), u = ilqr.U.reshape(horizon,control_dimension), horizon=horizon)
    run_fish.draw_figure(path_to_traj_fig)
