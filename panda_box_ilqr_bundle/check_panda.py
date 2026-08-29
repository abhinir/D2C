import mujoco
import numpy as np

from panda_model import PandaReachModel


model = PandaReachModel()

print("Model dimensions")
print("----------------")
print("nq =", model.nq)
print("nv =", model.nv)
print("nx =", model.nx)
print("nu =", model.nu)
print("physics dt =", model.dt_physics)
print("control dt =", model.dt)
print("horizon =", model.horizon)
print("final time =", model.horizon * model.dt)

print("\nActuators")
print("---------")
for i in range(model.nu):
    name = mujoco.mj_id2name(
        model.model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        i,
    )
    print(i, name)

print("\nInitial state")
print("-------------")
print("q0 =", model.q0)
print("qdot0 =", model.qd0)

print("\nTask")
print("----")
print("initial EE position =", model.end_effector_position(model.x0))
print("goal position       =", model.goal_pos)
print(
    "initial position error =",
    np.linalg.norm(model.position_error(model.x0)),
)

print("\nBox torque limits [Nm]")
print("----------------------")
print("u_min =", model.u_min)
print("u_max =", model.u_max)

u0 = model.initial_control_guess()
print("\nInitial gravity compensation [Nm]")
print("---------------------------------")
print(u0[:, 0])

x0_roll = model.rollout(u0)
print("\nInitial-guess trajectory cost =", model.trajectory_cost(x0_roll, u0))

np.savez(
    "panda_initial_guess.npz",
    x_trajectory=x0_roll,
    u_trajectory=u0,
)

print("\nSaved panda_initial_guess.npz")
