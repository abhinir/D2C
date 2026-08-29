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

from sim_car import SimulateCar
from car_params import *
from main_ilqr import iLQR
from ltv_sys_id import LTV_SysID

class RunCar(SimulateCar):

    def __init__(self, state_dimension, control_dimension, dt):
        SimulateCar.__init__(self, state_dimension, control_dimension, dt)

    def simulate_step(self,x,u):
        return self.simulate_trajectory(x, u.reshape((1, self.nu)))[-1]

class RunCarNoisy(SimulateCar):
    """CartPole with additive control noise"""
    def __init__(self, state_dimension, control_dimension, dt, noise_epsilon=0.0):
        SimulateCar.__init__(self, state_dimension, control_dimension, dt)
        self.noise_epsilon = noise_epsilon
    
    def simulate(self, x, u):
        # Add control noise: u += epsilon * 20 * N(0,1)
        if self.noise_epsilon > 0:
            noise = np.random.normal(0, 1, size=u.shape)
            u_noisy = u + self.noise_epsilon * 1.0 * noise
        else:
            u_noisy = u
        return self.simulate_trajectory(x, u_noisy.reshape((1, self.nu)))[-1]

if __name__=="__main__":

    cwd = os.getcwd()
    path_to_vdp = Path(cwd)/"examples/car-like"
    MODEL = path_to_vdp/"models/None.xml"

    path_to_export = path_to_vdp/"Car_Experiments/exp_1"
    path_to_policy_file = path_to_export/"car_policy.txt"
    path_to_cost_file = path_to_export / "training_cost_data.txt"
    path_to_training_cost_fig = path_to_export/"episodic_cost_training.png"
    path_to_traj_fig = path_to_export/"optimal_traj.png"
    # path_to_data = path_to_export / "vdp_D2C_data.txt"

    init_state = np.array([0.0, 0.0, np.pi/3, 0.0]).reshape((state_dimension,1))
    final_state = np.array([1.0, 4.0, np.pi/2, 0.0]).reshape((state_dimension, 1))

    print('Initial phase : \n', init_state)
    print('Goal phase : \n', final_state)
	
    # No. of ILQR iterations to run
    n_iterations = 100

    # Create model instance
    run_car = RunCar(state_dimension, control_dimension, dt)
    # run_car = RunCarNoisy(state_dimension, control_dimension, dt, noise_epsilon=0.0)
    u1 = np.load('examples/car-like/u_test.npy').T
    # Create iLQR instance
    ilqr = iLQR(run_car, state_dimension, control_dimension, alpha, horizon, init_state, final_state, Q, Q_final, R, 
                nominal_init_stddev, n_sys_id_samples=30, pert_sys_id_sigma=1e-7, arma_sys_id_flag = True)
    ilqr.iterate_ilqr(n_iterations)

    # print(ilqr.episodic_cost_history)

    ilqr.plot_episodic_cost_history(path_to_training_cost_fig)
    ilqr.save_policy(path_to_policy_file)
    ilqr.save_cost(path_to_cost_file)
    
    # Check and Simulate the obtained policy
    # print
    # run_car.simulate_car(state_init = init_state.flatten(), u = u1, horizon=horizon, n_per_step=1)
    # run_car.draw_figure(path_to_traj_fig)

    print(ilqr.U)
    run_car.simulate_trajectory(state_init = init_state.flatten(), u = ilqr.U.reshape((horizon,control_dimension)), horizon=horizon)
    run_car.draw_figure(path_to_traj_fig)

    # Test sys_id
    """ x_t = np.array([2.0,0.0]).reshape(state_dimension,1)
    u_t = np.array([0]).reshape(control_dimension,1)
    AB = model.sys_id(x_t,u_t)
    print(AB) """