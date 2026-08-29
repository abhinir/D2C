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

from sim_quadruped import SimulateGo2
from quadruped_params import *
from main_ilqr import iLQR
from ltv_sys_id import LTV_SysID

class RunGo2(SimulateGo2):

    def __init__(self, state_dimension, control_dimension, dt, model_path=None):
        SimulateGo2.__init__(self, state_dimension, control_dimension, dt, model_path=str(model_path))

    def simulate_step(self,x,u):
        return self.simulate_quadruped(x, u)
        # return self.simulate_trajectory(x,u.reshape(1,self.nu))[-1]


if __name__=="__main__":

    cwd = os.getcwd()
    path_to_quadruped = Path(cwd)/"examples/quadruped"
    MODEL = path_to_quadruped/"models/scene.xml"

    path_to_export = path_to_quadruped/"Quadruped_Experiments/exp_1"
    path_to_policy_file = path_to_export/"quadruped_policy.txt"
    path_to_cost_file = path_to_export / "training_cost_data.txt"
    path_to_training_cost_fig = path_to_export/"episodic_cost_training.png"
    path_to_traj_fig = path_to_export/"optimal_traj.png"
    
    init_state = np.zeros((state_dimension,1))
    init_state[2] = 0.27
    init_state[3] = 1
    init_state[7:19] = np.array([0, 0.9, -1.8] * 4).reshape((12,1))
    final_state = np.copy(init_state)
    # final_state[0] = 2.0

    # No. of ILQR iterations to run
    n_iterations = 10

    # Create model instance
    run_go2 = RunGo2(state_dimension, control_dimension, dt, MODEL)

    u_init = np.tile(np.array([
                -2.03926889,  0.59256921,  5.88984517,  2.03927658,  0.59256386,  5.88986376,
                -2.18573907,  0.55455663,  6.24895975,  2.18573223,  0.55455287,  6.24893569
                ]), (horizon, 1)).reshape((horizon, control_dimension, 1))
    

    print('Initial phase : \n', init_state)
    print('Goal phase : \n', final_state)

    # Create iLQR instance
    ilqr = iLQR(run_go2, state_dimension, control_dimension, alpha, horizon, init_state, final_state, Q, Q_final, R, 
                nominal_init_stddev, n_sys_id_samples=100, pert_sys_id_sigma=1e-2, arma_sys_id_flag = True)
    ilqr.iterate_ilqr(n_iterations,u_init=u_init)


    ilqr.plot_episodic_cost_history(path_to_training_cost_fig)
    ilqr.save_policy(path_to_policy_file)
    ilqr.save_cost(path_to_cost_file)

    # Check and Simulate the obtained policy
    run_go2.simulate_trajectory(y_init = init_state.flatten(), u = ilqr.U.reshape(horizon,control_dimension), horizon=horizon)
    run_go2.draw_figure(path_to_traj_fig)

    # Test sys_id
    '''x_t = np.array([2.0,0.0]).reshape(state_dimension,1)
    u_t = np.array([0]).reshape(control_dimension,1)
    AB = model.sys_id(x_t,u_t)
    print(AB)'''