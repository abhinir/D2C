from iLQR_model_free import ModelFree_ILQR
from box_iLQR_model_free import ModelFree_Box_ILQR
from models import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mujoco
import mujoco.viewer
import imageio


barr_param = 1e6
red_factor = 0.8
beta = 1/0.99
outer_loop_iter = 1
#model = Pendulum_Mujoco()
model = PandaReachModel()
#model = Pendulum(int_type = "RK4")


#control = np.loadtxt('data/controls_u.csv', delimiter=',')
#control = 0*np.ones((model.nu, model.horizon))
control = model.initial_control_guess()

for i in range(outer_loop_iter):
    if barr_param < 1e-3:
        break
        
    if red_factor > 1:
        print(f"Optimization Stopped at barrier parameter:{barr_param}")
        break
    
    box_ilqr = ModelFree_Box_ILQR(
        model,
        max_iterations=1000,
        alpha = 1,
        sigma = barr_param,
        verbose=True
    )
    
    #ilqr.main_func(u_init=np.vstack([model.g*np.ones((1, model.horizon)),np.zeros((model.nu-1, model.horizon))]), x0 = model.x0)
    box_ilqr.main_func(u_init = control, x0 = model.x0)
    print(f"Minimum control: {np.min(box_ilqr.u_optimal)}")
    print(f"Maximum control: {np.max(box_ilqr.u_optimal)}")
#    for m in range(model.horizon):
#        constraint_violation, indices =  model.check_constraint_violation(box_ilqr.x_optimal[:,m], box_ilqr.u_optimal[:,m])
#        if constraint_violation & (m == model.horizon - 1):
#            control = box_ilqr.u_optimal
#            barr_param = barr_param*red_factor
#            break
#        elif not constraint_violation:
#            barr_param = barr_param/red_factor
#            red_factor = red_factor*beta
#            barr_param = barr_param*red_factor
#            break
#        else:
#            continue
    



    



np.savez(
    "panda_solution.npz",
    x_trajectory=box_ilqr.x_optimal,
    u_trajectory=box_ilqr.u_optimal
)
