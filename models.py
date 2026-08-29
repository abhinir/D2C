from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from Models import *
import mujoco
import time
from pathlib import Path



def propagate_forward(model, curr_state, control, t):
    if model.int_type == "euler":
        temp = curr_state
        for i in range(model.steps):
            temp = temp + model.eom(temp, control, t)*model.dt/(model.steps)

        x_out = temp
        
    elif model.int_type == "RK4":
        fun = lambda tau, x: model.eom(x, control, t)
        sol = solve_ivp(fun, [t, t+model.dt], curr_state, method='RK45')
        
        x_out = sol.y[:, -1]
        
    return np.array(x_out).T
            
class Pendulum():
    def __init__(self, int_type = 'euler', steps = 1000, u_max = 1, u_min = -1, \
                 x0 = np.array([0,0]), \
                 xg = np.array([np.pi, 0]), dt = 0.1, horizon = 50):
        self.name = 'pendulum'
        self.m = 0.5
        self.L = 0.5
        self.g = 9.81
        
        self.nx = 2                                     #no. of states
        self.nu = 1                                     #no. of control inputs
        self.dt = dt
        self.horizon = horizon                              #time-step
        self.int_type = int_type                        #type of solver used to integrate dynamics
        self.steps = steps                                #no. of steps of the integrator
        self.Q = 3*np.diag(np.array([1,1]))          #incremental state penalty
        self.R = 3                                      #Incremental cost penalty
        self.Qf = 30*np.eye(self.nx)                   #Quadratic penalty fro terminal cost
        self.u_max = u_max                              #Maximum value of control
        self.u_min = u_min                              #Minimum value of control
        self.x0 = x0                                    #Initial State
        self.xg = xg                                    #Goal State

    def eom(self, x, u, t):
        """
        Computes the equations of motion for a controlled simple pendulum.

        Parameters:
            state : array-like, [theta, theta_dot]
            control : float, external torque applied to the pendulum

        Returns:
            np.ndarray : [theta_dot, theta_double_dot]
        """
#        print(len(x))
#        assert isinstance(x, (list, np.ndarray)) and len(x) == 2, "State must be [theta, theta_dot]"
#        assert np.isscalar(u), "Control must be a scalar torque"
        u = u.item()
        theta, theta_dot = x
        theta_double_dot = (-self.g * np.sin(theta) / self.L) + (u / (self.m * self.L ** 2))
        return np.array([theta_dot, theta_double_dot]).T
    
    def propagate_dynamics(self, x, u, t):
        next_state = propagate_forward(self, x, u, t)
        return next_state
        
    def cost(self, x, u, cost_type="quadratic", sigma=100):
        """
        Compute the incremental cost for a given state x and control u.

        Parameters:
            x : np.ndarray or float
                Current state
            u : np.ndarray or float
                Control input
            cost_type : str
                Determine cost form ("quadratic" or "log-barrier")
            sigma : float
                Barrier parameter (for log-barrier version)

        Returns:
            cost : float
                Incremental cost
        """
        # Quadratic cost
        state_error = x - self.xg
        quadratic_state_cost = 0.5 * state_error.T @ self.Q @ state_error * self.dt
        
        
        quadratic_control_cost = 0.5 * u * self.R * u * self.dt

        cost = quadratic_state_cost + quadratic_control_cost

        # Log-barrier cost
        if cost_type != "quadratic":
            # Avoid log(0) errors
            eps = 1e-8
            u_min = self.u_min
            u_max = self.u_max
            cost -= sigma * np.log(u - u_min, eps)
            cost -= sigma * np.log(u_max - u, eps)

        return cost

    def terminal_cost(self, x):
        state_error = x - self.xg
        terminal_cost = 0.5 * state_error.T @ self.Qf @ state_error
        
        return terminal_cost
            
        
        
class Cartpole():
    def __init__(self, int_type = 'euler', steps = 10000, u_max = 2, u_min = -2, \
                 x0 = np.array([0, 0, np.pi, 0]), \
                 xg = np.array([0, 0, 0, 0]), dt = 0.01, horizon = 300):
        self.name = 'cartpole'
        self.M = 1
        self.m = 0.01
        self.L = 0.6
        self.g = 9.81
        
        self.nx = 4                                     #no. of states
        self.nu = 1                                     #no. of control inputs
        self.dt = dt                                    #time-step
        self.horizon = horizon
        self.int_type = int_type                        #type of solver used to integrate dynamics
        self.steps = steps                                #no. of steps of the integrator
        self.Q = 10*np.diag(np.array([1,1,1,1]))          #incremental state penalty
        self.R = 10                                      #Incremental cost penalty
        self.Qf = 1e4*np.eye(self.nx)                   #Quadratic penalty for terminal cost
        self.u_max = u_max                              #Maximum value of control
        self.u_min = u_min                              #Minimum value of control
        self.x0 = x0                                    #Initial State
        self.xg = xg                                    #Goal State

    def eom(self, x, u, t):
        """
        Computes the equations of motion for a controlled simple pendulum.

        Parameters:
            state : array-like, [theta, theta_dot]
            control : float, external torque applied to the pendulum

        Returns:
            np.ndarray : [theta_dot, theta_double_dot]
        """
#        print(len(x))
#        assert isinstance(x, (list, np.ndarray)) and len(x) == 2, "State must be [theta, theta_dot]"
#        assert np.isscalar(u), "Control must be a scalar torque"
        u = u.item()
        x_pos, x_dot, theta, theta_dot = x
        T1 = self.m/(self.M + self.m - self.m*(np.cos(theta)**2))
        T2 = T1*(self.g*np.sin(theta)*np.cos(theta) - self.L*np.sin(theta)*theta_dot**2)
        x_double_dot = T2 + T1*u/self.m
        theta_double_dot = (self.g * np.sin(theta) / self.L) + (np.cos(theta)/self.L)*(T2+ T1*u/self.m)
        return np.array([x_dot, x_double_dot, theta_dot, theta_double_dot]).T
    
    def propagate_dynamics(self, x, u, t):
        next_state = propagate_forward(self, x, u, t)
        return next_state
        
    def cost(self, x, u, cost_type="quadratic", sigma=100):
        """
        Compute the incremental cost for a given state x and control u.

        Parameters:
            x : np.ndarray or float
                Current state
            u : np.ndarray or float
                Control input
            cost_type : str
                Determine cost form ("quadratic" or "log-barrier")
            sigma : float
                Barrier parameter (for log-barrier version)

        Returns:
            cost : float
                Incremental cost
        """
        # Quadratic cost
        state_error = x - self.xg
        quadratic_state_cost = 0.5 * state_error.T @ self.Q @ state_error * self.dt
        
        
        quadratic_control_cost = 0.5 * u * self.R * u * self.dt

        cost = quadratic_state_cost + quadratic_control_cost

        # Log-barrier cost
        if cost_type != "quadratic":
            # Avoid log(0) errors
            eps = 1e-8
            u_min = self.u_min
            u_max = self.u_max
            cost -= sigma * np.log(u - sel.u_min, eps)
            cost -= sigma * np.log(self.u_max - u, eps)

        return cost

    def terminal_cost(self, x):
        state_error = x - self.xg
        terminal_cost = 0.5 * state_error.T @ self.Qf @ state_error
        
        return terminal_cost
            
class Acrobot():
    def __init__(self, int_type = 'euler', steps = 10, u_max = 3, u_min = -3, \
                 x0 = np.array([0,0,0,0]), \
                 xg = np.array([np.pi, 0, 0, 0]), dt = 0.001, horizon = 10000):
        self.name = 'acrobot'
        self.m1 = 0.5
        self.m2 = 0.5
        self.l1 = 1
        self.lc1 = 0.5
        self.I1 = 0.5*self.m1*(self.l1**2)/12
        self.l2 = 1
        self.g = 9.81
        self.lc2 = 0.5
        self.I2 = 0.5*self.m2*(self.l2**2)/12
        
        self.nx = 4                                     #no. of states
        self.nu = 1                                     #no. of control inputs
        self.dt = dt
        self.horizon = horizon                              #time-step
        self.int_type = int_type                        #type of solver used to integrate dynamics
        self.steps = steps                                #no. of steps of the integrator
        self.Q = 10*np.diag(np.array([1,1,1,1]))          #incremental state penalty
        self.R = 10                                      #Incremental cost penalty
        self.Qf = 1e4*np.eye(self.nx)                   #Quadratic penalty fro terminal cost
        self.u_max = u_max                              #Maximum value of control
        self.u_min = u_min                              #Minimum value of control
        self.x0 = x0                                    #Initial State
        self.xg = xg                                    #Goal State

    def eom(self, x, u, t):
       # --- Model Parameters ---
        l1 = self.l1
        m1 = self.m1
        lc1 = self.lc1
        I1 = self.I1
        l2 = self.l2
        m2 = self.m2
        lc2 = self.lc2
        I2 = self.I2
        g = self.g

        # --- Unpack State Vector ---
        th1, th2, th1_dot, th2_dot = x

        
        # --- Kinematic Equations ---
        # The first two elements of the derivative are the velocities
        state_dot = np.zeros(4)
        state_dot[0] = th1_dot
        state_dot[1] = th2_dot

        # --- Dynamics Matrices (from Euler-Lagrange) ---
        c2 = np.cos(th2)
        s2 = np.sin(th2)
        s1 = np.sin(th1)
        s12 = np.sin(th1 + th2)
        
        # M(q) - Mass/Inertia Matrix
        M = np.array([
            [I1 + I2 + m2 * (l1**2) + 2 * m2 * l1 * lc2 * c2, I2 + m2 * l1 * lc2 * c2],
            [I2 + m2 * l1 * lc2 * c2,                          I2]
        ])

        # C(q, q_dot) - Coriolis and Centrifugal Matrix
        C = np.array([
            [-2 * m2 * l1 * lc2 * s2 * th2_dot, -m2 * l1 * lc2 * s2 * th2_dot],
            [m2 * l1 * lc2 * s2 * th1_dot,      0]
        ])

        # G(q) - Gravity Vector
        G = np.array([
            (m1 * lc1 + m2 * l1) * g * s1 + m2 * lc2 * g * s12,
            m2 * lc2 * g * s12
        ])
        
        # B - Input Matrix
        B = np.array([0, 1])
        
        # --- Solve for Accelerations (q_ddot) ---
        q_dot = np.array([th1_dot, th2_dot])
        
        # RHS of the equation: M*q_ddot = (B*U - C*q_dot - G)
        rhs = B * u - C @ q_dot - G
        
        # Solve the linear system M * q_ddot = rhs for q_ddot
        q_ddot = np.linalg.inv(M)@rhs
        
        # --- Assign Accelerations to State Derivative ---
        state_dot[2] = q_ddot[0]
        state_dot[3] = q_ddot[1]

        return state_dot
    
    def propagate_dynamics(self, x, u, t):
        next_state = propagate_forward(self, x, u, t)
        return next_state
        
    def cost(self, x, u, cost_type="quadratic", sigma=100000):
        """
        Compute the incremental cost for a given state x and control u.

        Parameters:
            x : np.ndarray or float
                Current state
            u : np.ndarray or float
                Control input
            cost_type : str
                Determine cost form ("quadratic" or "log-barrier")
            sigma : float
                Barrier parameter (for log-barrier version)

        Returns:
            cost : float
                Incremental cost
        """
        # Quadratic cost
        
        state_error = x - self.xg
        quadratic_state_cost = 0.5 * state_error.T @ self.Q @ state_error * self.dt
        
#        u = u.item()
        quadratic_control_cost = 0.5 * u * self.R * u * self.dt

        cost = quadratic_state_cost + quadratic_control_cost

        # Log-barrier cost
#        if cost_type != "quadratic":
            # Avoid log(0) errors
        eps = 1e-8
        u_min = self.u_min
        u_max = self.u_max
        cost -= sigma * np.log(u - self.u_min)
        cost -= sigma * np.log(self.u_max - u)

        return cost

    def terminal_cost(self, x):
        state_error = x - self.xg
        terminal_cost = 0.5 * state_error.T @ self.Qf @ state_error
        
        return terminal_cost


class Car_linear():
    def __init__(self, int_type = 'euler', steps = 1, u_max = 1, u_min = -1, \
                 x0 = np.array([0,0,0,0]), \
                 xg = np.array([10, 0, 0, 10]), dt = 1, horizon = 20):
        self.name = 'car_linear'
        self.m = 0.5
        self.L = 0.5
        self.g = 9.81
        
        self.nx = 4                                     #no. of states
        self.nu = 4                                     #no. of control inputs
        self.dt = dt
        self.horizon = horizon                              #time-step
        self.int_type = int_type                        #type of solver used to integrate dynamics
        self.steps = steps                                #no. of steps of the integrator
        self.Q = 3*np.diag(np.array([1,1]))          #incremental state penalty
        self.R = 1e-2*np.diag(np.array([1,1,1,1]))    #Incremental cost penalty
        self.Qf = 30*np.eye(self.nx)                   #Quadratic penalty fro terminal cost
        self.u_max = u_max                              #Maximum value of control
        self.u_min = u_min                              #Minimum value of control
        self.x0 = x0                                    #Initial State
        self.xg = xg                                    #Goal State

    def eom(self, x, u, t):
        """
        Computes the equations of motion for a controlled simple pendulum.

        Parameters:
            state : array-like, [theta, theta_dot]
            control : float, external torque applied to the pendulum

        Returns:
            np.ndarray : [theta_dot, theta_double_dot]
        """
        return np.array([u[0], u[2], u[2], u[3]]).T
    
    def propagate_dynamics(self, x, u, t):
        next_state = propagate_forward(self, x, u, t)
        return next_state
        
    def cost(self, x, u, x_prev, cost_type="quadratic", sigma=100):
        """
        Compute the incremental cost for a given state x and control u.

        Parameters:
            x : np.ndarray or float
                Current state
            u : np.ndarray or float
                Control input
            cost_type : str
                Determine cost form ("quadratic" or "log-barrier")
            sigma : float
                Barrier parameter (for log-barrier version)

        Returns:
            cost : float
                Incremental cost
        """
        # Quadratic cost
        state_error = np.linalg.norm(x - self.xg, 2)
        state_error_prev = np.linalg.norm(x_prev - self.xg, 2)
        obs_cost = np.linalg.norm(x[0:1] - x_prev[2:3], 2)
        obs_cost_prev = np.linalg.norm(x_prev[0:1] - x_prev[2:3], 2)
        
        control_cost = np.exp(10*(abs(u[0]) - 1)) + np.exp(10*(abs(u[1]) - 1)) + np.exp(10*(abs(u[2]) - 1)) + np.exp(10*(abs(u[3]) - 1))
        cost = 1*(state_error - state_error_prev) - 0.75*(obs_cost - obs_cost_prev) + control_cost
        
        return cost
    def terminal_cost(self, x):
        return 0
        

class Car_linear():
    def __init__(self, int_type = 'euler', steps = 1, u_max = 1, u_min = -1, \
                 x0 = np.array([0,0,0,0]), \
                 xg = np.array([10, 0, 0, 10]), dt = 1, horizon = 20):
        self.name = 'car_linear'
        self.m = 0.5
        self.L = 0.5
        self.g = 9.81
        
        self.nx = 4                                     #no. of states
        self.nu = 4                                     #no. of control inputs
        self.dt = dt
        self.horizon = horizon                              #time-step
        self.int_type = int_type                        #type of solver used to integrate dynamics
        self.steps = steps                                #no. of steps of the integrator
        self.Q = 3*np.diag(np.array([1,1]))          #incremental state penalty
        self.R = 1e-2*np.diag(np.array([1,1,1,1]))    #Incremental cost penalty
        self.Qf = 30*np.eye(self.nx)                   #Quadratic penalty fro terminal cost
        self.u_max = u_max                              #Maximum value of control
        self.u_min = u_min                              #Minimum value of control
        self.x0 = x0                                    #Initial State
        self.xg = xg                                    #Goal State

    def eom(self, x, u, t):
        """
        Computes the equations of motion for a controlled simple pendulum.

        Parameters:
            state : array-like, [theta, theta_dot]
            control : float, external torque applied to the pendulum

        Returns:
            np.ndarray : [theta_dot, theta_double_dot]
        """
        return np.array([u[0], u[2], u[2], u[3]]).T
    
    def propagate_dynamics(self, x, u, t):
        next_state = propagate_forward(self, x, u, t)
        return next_state
        
    def cost(self, x, u, x_prev, cost_type="quadratic", sigma=100):
        """
        Compute the incremental cost for a given state x and control u.

        Parameters:
            x : np.ndarray or float
                Current state
            u : np.ndarray or float
                Control input
            cost_type : str
                Determine cost form ("quadratic" or "log-barrier")
            sigma : float
                Barrier parameter (for log-barrier version)

        Returns:
            cost : float
                Incremental cost
        """
        # Quadratic cost
        state_error = np.linalg.norm(x - self.xg, 2)
        state_error_prev = np.linalg.norm(x_prev - self.xg, 2)
        obs_cost = np.linalg.norm(x[0:1] - x_prev[2:3], 2)
        obs_cost_prev = np.linalg.norm(x_prev[0:1] - x_prev[2:3], 2)
        
        control_cost = np.exp(10*(abs(u[0]) - 1)) + np.exp(10*(abs(u[1]) - 1)) + np.exp(10*(abs(u[2]) - 1)) + np.exp(10*(abs(u[3]) - 1))
        cost = 1*(state_error - state_error_prev) - 0.75*(obs_cost - obs_cost_prev) + control_cost
        
        return cost
    def terminal_cost(self, x):
        return 0
        
class Quadrotor():
    def __init__(self, int_type = 'euler', steps = 1, u_max = 1, u_min = -1, \
                 x0 = np.array([0.2,2.5,2.5,0,0,0]), \
                 xg = np.array([4.0, 2.5, 2.5, 0, 0, 0]), dt = 0.01, horizon = 500):
        self.name = 'Quadrotor'
        self.m = 0.5
        self.L = 0.5
        self.g = 9.81
        
        self.nx = 6                                     #no. of states
        self.nu = 3                                     #no. of control inputs
        self.dt = dt
        self.horizon = horizon                              #time-step
        self.int_type = int_type                        #type of solver used to integrate dynamics
        self.steps = steps                                #no. of steps of the integrator
        self.Q = 300*np.diag(np.array([1,1,1,1,1,1]))          #incremental state penalty
        self.R = 1e2*np.diag(np.array([1,2.5,2.5]))    #Incremental cost penalty
        self.Qf = 15000*np.eye(self.nx)                   #Quadratic penalty fro terminal cost
        self.u_max = u_max                              #Maximum value of control
        self.u_min = u_min                              #Minimum value of control
        self.x0 = x0                                    #Initial State
        self.xg = xg                                    #Goal State

    def eom(self, x, u, t):
        """
        Computes the equations of motion for a controlled simple pendulum.

        Parameters:
            state : array-like, [theta, theta_dot]
            control : float, external torque applied to the pendulum

        Returns:
            np.ndarray : [theta_dot, theta_double_dot]
        """
        x_dot = np.zeros_like(x)
        x_dot[...,0] = x[...,3]
        x_dot[...,1] = x[...,4]
        x_dot[...,2] = x[...,5]
        x_dot[...,3] = self.g * np.tan(u[...,2])
        x_dot[...,4] = -self.g * np.tan(u[...,1])
        x_dot[...,5] = u[...,0] - self.g
        return x_dot
    
    def propagate_dynamics(self, x, u, t):
        next_state = propagate_forward(self, x, u, t)
        return next_state
        
    def cost(self, x, u, cost_type="quadratic", sigma=100):
        """
        Compute the incremental cost for a given state x and control u.

        Parameters:
            x : np.ndarray or float
                Current state
            u : np.ndarray or float
                Control input
            cost_type : str
                Determine cost form ("quadratic" or "log-barrier")
            sigma : float
                Barrier parameter (for log-barrier version)

        Returns:
            cost : float
                Incremental cost
        """
        # Quadratic cost
        obs = np.array([[2.0, 1.5, 0.5], [2.0, 0.5, 2.5], [2.0, 1.5, 4.5], [2.0, 3.5, 0.5], [2.0, 2.5, 2.5], [2.0, 3.5, 4.5], [2.0, 4.5, 2.5]])
#        state_error = np.linalg.norm(x[...,:3] - self.xg[...,:3])

        obs_cost = 0
        dist = np.inf
        for i in range(7):
#            dist = np.minimum(dist, np.linalg.norm(x[...,:3] - obs[i,:]) - 0.5)
            dist = np.linalg.norm(x[...,:3] - obs[i,:]) - 0.5
            obs_cost = obs_cost + 0.0001*np.exp(-0.0001*(dist))
#        dist_to_bound = np.min(np.stack([
#            x[..., 0],
#            x[..., 1],
#            x[..., 2],
#            5 - x[..., 0],
#            5 - x[..., 1],
#            5 - x[..., 2]
#        ]))
#        dist = np.minimum(dist, dist_to_bound)
#
#        obs_cost = obs_cost + 2*np.exp(-2*(dist_to_bound))
##
#        bound_cost = 0
#        for i in range(6):
#            dist_to_bound = np.minimum((np.minimum(x[...,0],np.abs(x[...,0] - 5)), np.minimum(x[...,1],np.abs(x[...,1] - 5)), np.minimum(x[...,2],np.abs(x[...,2] - 5))))
#            bound_cost = bound_cost + 10*np.exp(-10*dist_to_bound)
        state_cost = 0.5*(x - self.xg) @ self.Q @ (x - self.xg)
        control_cost = 0.5*(u - [self.g, 0, 0]).T @ self.R @ (u - [self.g, 0, 0])
#        state_cost = (x - self.xg).T @ self.Q @ (x - self.xg)
#        control_penalty = 5*np.exp(5*(abs(u[0] - self.g)-1)) + 2*np.exp(5*(abs(u[1]) - 0.4)) + 2*np.exp(5*(abs(u[2]) - 0.4))

        cost = (state_cost) + obs_cost + control_cost
        
        return cost
    def terminal_cost(self, x):
#        state_error = np.linalg.norm(x[...,:3] - self.xg[...,:3], 2)
        terminal_cost =  0.5*(x - self.xg) @ self.Qf @ (x - self.xg)
        return terminal_cost
    

class Walker():
    def __init__(self, int_type = 'euler', steps = 1, u_max = 10, u_min = -10, \
             x0 = np.zeros((18)), \
             xg = np.zeros((18)), dt = 0.05, horizon = 100):
        self.name = 'Walker'
        xml_path = "Models/walker.xml"
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.nx = 18                                    #no. of states
        self.nu = 6                                     #no. of control inputs
        self.dt = dt
        self.horizon = horizon                          #time-step
        self.int_type = int_type                        #type of solver used to integrate dynamics
        self.steps = steps                              #no. of steps of the integrator
        self.Q = 30*np.eye(self.nx)                     #incremental state penalty
        self.R = 1*np.eye(self.nu)                      #Incremental cost penalty
        self.Qf = 30*np.eye(self.nx)                   #Quadratic penalty fro terminal cost
        self.u_max = u_max                              #Maximum value of control
        self.u_min = u_min                              #Minimum value of control
        self.x0 = np.concatenate((self.data.qpos, self.data.qvel))                                    #Initial State
#        self.x0 = x0
#        self.xg = self.x0                                    #Goal State
        target_qpos_athletic = np.array([
            1.3, # rootz: height
            0.0,  # rootx: position
            0.0,  # rooty: torso pitch
            0.0,  # right_hip
            0.0,  # right_knee
            0.0,  # right_ankle
            0.0,  # left_hip
            0.0,  # left_knee
            0.0,  # left_ankle
        ])


        # For a stationary target, all velocities are zero
        target_qvel = np.zeros((9))
        
        self.xg = np.concatenate((target_qpos_athletic, target_qvel))
        
    
    def propagate_dynamics(self, x, u, t):
        
        self.data.qpos = x[:9]
        self.data.qvel = x[9:]
        for i in range(100):
            self.data.ctrl = u
            mujoco.mj_step(self.model, self.data)
        return np.concatenate((self.data.qpos, self.data.qvel))
        
    def cost(self, x, u, cost_type="quadratic", sigma=100):
        """
        Compute the incremental cost for a given state x and control u.

        Parameters:
            x : np.ndarray or float
                Current state
            u : np.ndarray or float
                Control input
            cost_type : str
                Determine cost form ("quadratic" or "log-barrier")
            sigma : float
                Barrier parameter (for log-barrier version)

        Returns:
            cost : float
                Incremental cost
        """
        state_error = x - self.xg
        
        quadratic_state_cost = 0.5 * state_error.T @ self.Q @ state_error * self.dt
        quadratic_control_cost = 0.5 * u.T @ self.R @ u * self.dt
        
        cost = quadratic_state_cost + quadratic_control_cost
        
        
        return cost
        
    def terminal_cost(self, x):
        state_error = x - self.xg
        terminal_cost = 0.5 * state_error.T @ self.Qf @ state_error
        
        return terminal_cost
    
class Pendulum_Mujoco():
    def __init__(self, int_type = 'euler', steps = 1, u_max = 2, u_min = -2, \
             x0 = np.array([0,0]), \
             xg = np.array([np.pi, 0]), dt = 0.1, horizon = 50):
        self.name = 'Pendulum_Mujoco'
        xml_path = "Models/pendulum.xml"
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.nx = 2                                     #no. of states
        self.nu = 1                                     #no. of control inputs
        self.time_steps = 5
        self.dt = self.model.opt.timestep*self.time_steps
        self.horizon = horizon                          #time-step
        self.int_type = int_type                        #type of solver used to integrate dynamics
        self.steps = steps                              #no. of steps of the integrator
        self.Q = 3*np.eye(self.nx)                    #incremental state penalty
        self.R = 3*np.eye(self.nu)                    #Incremental cost penalty
        self.Qf = 30*np.eye(self.nx)                 #Quadratic penalty fro terminal cost
        self.u_max = u_max                              #Maximum value of control
        self.u_min = u_min                              #Minimum value of control
        self.x0 = np.concatenate((self.data.qpos, self.data.qvel))                                    #Initial State
        self.xg =  xg                                   #Goal State
        self.xml_path = "Models/pendulum.xml"
        self.video_duration = 5
        self.framerate = 50
        self.output_path = "pedulum_swing_up.mp4"
    
    def propagate_dynamics(self, x, u, t):
        self.data.ctrl = u
        self.data.qpos = x[0]
        self.data.qvel = x[1]
        for i in range(self.time_steps):
            mujoco.mj_step(self.model, self.data)
        return np.concatenate((self.data.qpos, self.data.qvel))
        
    def cost(self, x, u, sigma, cost_type="quadratic"):
        """
        Compute the incremental cost for a given state x and control u.

        Parameters:
            x : np.ndarray or float
                Current state
            u : np.ndarray or float
                Control input
            cost_type : str
                Determine cost form ("quadratic" or "log-barrier")
            sigma : float
                Barrier parameter (for log-barrier version)

        Returns:
            cost : float
                Incremental cost
        """
        state_error = x - self.xg
        
        quadratic_state_cost = 0.5 * state_error.T @ self.Q @ state_error * self.dt
        quadratic_control_cost = 0.5 * u.T @ self.R @ u * self.dt
        barrier_cost = self.dt*(np.log(self.u_max - u) + np.log(u - self.u_min))
        cost = quadratic_state_cost + quadratic_control_cost - barrier_cost*sigma
        
        
        return float(np.asarray(cost).squeeze())
        
    def terminal_cost(self, x):
        state_error = x - self.xg
        terminal_cost = 0.5 * state_error.T @ self.Qf @ state_error
        
        return terminal_cost
    
class Biped():
    def __init__(self, int_type = 'euler', steps = 1, u_max = 10, u_min = -10, \
             x0 = np.zeros((18)), \
             xg = np.zeros((18)), dt = 0.05, horizon = 500):
        self.name = 'Biped'
        xml_path = "Models/biped.xml"
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.nx = 25                                    #no. of states
        self.nu = 6                                     #no. of control inputs
        self.dt = self.model.opt.timestep
        self.horizon = horizon                          #time-step
        self.int_type = int_type                        #type of solver used to integrate dynamics
        self.steps = steps                              #no. of steps of the integrator
        self.Q = np.diag([1,1,0,0.1,0.1,0.1,0.1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0])                     #incremental state penalty
        self.R = 1*np.eye(self.nu)                      #Incremental cost penalty
        self.Qf = self.Q                   #Quadratic penalty fro terminal cost
        self.u_max = u_max                              #Maximum value of control
        self.u_min = u_min                              #Minimum value of control
        self.x0 = np.concatenate((self.data.qpos, self.data.qvel))                                    #Initial State
        self.xg = self.x0                                    #Goal State

        
    
    def propagate_dynamics(self, x, u, t):
        
        self.data.qpos = x[:13]
        self.data.qvel = x[13:]
#        for i in range(100):
        self.data.ctrl = u
        mujoco.mj_step(self.model, self.data)
        return np.concatenate((self.data.qpos, self.data.qvel))
        
    def cost(self, x, u, sigma, cost_type="quadratic"):
        """
        Compute the incremental cost for a given state x and control u.

        Parameters:
            x : np.ndarray or float
                Current state
            u : np.ndarray or float
                Control input
            cost_type : str
                Determine cost form ("quadratic" or "log-barrier")
            sigma : float
                Barrier parameter (for log-barrier version)

        Returns:
            cost : float
                Incremental cost
        """
        state_error = x - self.xg
        
        quadratic_state_cost = 0.5 * state_error.T @ self.Q @ state_error * self.dt
        quadratic_control_cost = 0.5 * u.T @ self.R @ u * self.dt
        
        cost = quadratic_state_cost + quadratic_control_cost
        
        
        return float(np.asarray(cost).squeeze())
        
    def terminal_cost(self, x):
        state_error = x - self.xg
        terminal_cost = 0.5 * state_error.T @ self.Qf @ state_error
        
        return terminal_cost


class Quadruped():
    def __init__(self, int_type = 'euler', steps = 1, u_max = 10, u_min = -10, \
             x0 = np.zeros((18)), \
             xg = np.zeros((18)), dt = 0.05, horizon = 100):
        self.name = 'Quadruped'
        xml_path = "Models/scene.xml"
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.nx = 37                                    #no. of states
        self.nu = 12                                     #no. of control inputs
        self.dt = self.model.opt.timestep*5
        self.horizon = horizon                          #time-step
        self.int_type = int_type                        #type of solver used to integrate dynamics
        self.steps = steps                              #no. of steps of the integrator
        self.Q = 10*np.eye(self.nx)
        self.Q[0,0] = 1000                     #incremental state penalty
        self.R = 1*np.eye(self.nu)                      #Incremental cost penalty
        self.Qf = 100*self.Q
        self.Qf[0,0] = 100000                  #Quadratic penalty fro terminal cost
        self.u_max = u_max                              #Maximum value of control
        self.u_min = u_min                              #Minimum value of control
        self.x0 = np.concatenate((self.data.qpos, self.data.qvel))                                    #Initial State
        self.xg = self.x0                                    #Goal State

        
    
    def propagate_dynamics(self, x, u, t):
        if t == 0:
            mujoco.mj_resetData(self.model, self.data)
#            mujoco.mj_forward(self.model, self.data)
        
        self.data.qpos = x[:19]
        self.data.qvel = x[19:]
        
        for i in range(5):
            self.data.ctrl = u
            mujoco.mj_step(self.model, self.data)
        return np.concatenate((self.data.qpos, self.data.qvel))
        
    def cost(self, x, u, cost_type="quadratic", sigma=100):
        """
        Compute the incremental cost for a given state x and control u.

        Parameters:
            x : np.ndarray or float
                Current state
            u : np.ndarray or float
                Control input
            cost_type : str
                Determine cost form ("quadratic" or "log-barrier")
            sigma : float
                Barrier parameter (for log-barrier version)

        Returns:
            cost : float
                Incremental cost
        """
        state_error = x - self.xg
        
        quadratic_state_cost = 0.5 * state_error.T @ self.Q @ state_error * self.dt
        quadratic_control_cost = 0.5 * u.T @ self.R @ u * self.dt
        
        cost = quadratic_state_cost + quadratic_control_cost
        
        
        return cost
        
    def terminal_cost(self, x):
        state_error = x - self.xg
        terminal_cost = 0.5 * state_error.T @ self.Qf @ state_error
        
        return terminal_cost


class Fish():
    def __init__(self, int_type = 'euler', steps = 1, u_max = np.ones((5,1)), u_min = -np.ones((5,1)), \
             x0 = np.zeros((27)), \
             xg = np.zeros((27)), dt = 0.2, horizon = 500):
        self.name = 'Fish'
        self.xml_path = "Models/fish.xml"
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.nx = 27                                    #no. of states
        self.nu = 5                                     #no. of control inputs
        self.time_steps = 50
        self.dt = self.model.opt.timestep*self.time_steps
        self.horizon = horizon                          #time-step
        self.int_type = int_type                        #type of solver used to integrate dynamics
        self.steps = steps                              #no. of steps of the integrator
        
        self.mouth_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "mouth"
        )
        self.target_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "target"
        )
        
        self.cost_data = mujoco.MjData(self.model)
        self.Q = 100.0 * np.eye(3)

        self.R = 1e16*np.diag([
            1e-3,   # tail
            5e-3,   # tail twist
            5e-3,   # fins flap
            1e-2,   # left fin pitch
            1e-2    # right fin pitch
        ])

        self.Qf = 5000.0 * np.eye(3)

        self.Vf = 10.0 * np.eye(3)
        self.Wf = 1.0 * np.eye(3)

        self.w_omega = 1e-2
        self.u_max = u_max                              #Maximum value of control
        self.u_min = u_min                              #Minimum value of control
        self.x0 = np.concatenate((self.data.qpos, self.data.qvel))                                    #Initial State
        self.goal_pos = self.cost_data.geom_xpos[
            self.target_id
        ].copy()
        self.video_duration = 10
        self.output_path = "fish_unconstrained.mp4"
        self.framerate = 100

        
    
    def propagate_dynamics(self, x, u, t=None):

        # Reset all MuJoCo internal data
        if t==0:
            mujoco.mj_resetData(self.model, self.data)

        # Set state
        self.data.qpos[:] = x[:14]
        self.data.qvel[:] = x[14:]

        # Set control
        self.data.ctrl[:] = u

        # Update derived MuJoCo quantities
        mujoco.mj_forward(self.model, self.data)

        # Hold the same control for 50 physics steps
        for _ in range(self.time_steps):
            mujoco.mj_step(self.model, self.data)

        # Return next state
        x_next = np.concatenate((
            self.data.qpos.copy(),
            self.data.qvel.copy()
        ))

        return x_next
        
    def cost(self, x, u):

        nq = self.model.nq
        nv = self.model.nv

        # Load supplied state into separate MuJoCo data object
        self.cost_data.qpos[:] = x[:nq]
        self.cost_data.qvel[:] = x[nq:nq + nv]

        mujoco.mj_forward(
            self.model,
            self.cost_data
        )

        # Mouth position
        mouth_pos = self.cost_data.geom_xpos[
            self.mouth_id
        ].copy()

        # Position error
        error = mouth_pos - self.goal_pos

        # Body angular velocity
        # qvel = [vx vy vz wx wy wz internal_joint_rates...]
        omega = self.cost_data.qvel[3:6]

        position_cost = 0.5 * error.T @ self.Q @ error

        control_cost = 0.5 * u.T @ self.R @ u

        angular_velocity_cost = (
            0.5 * self.w_omega * omega.T @ omega
        )

        return (
            position_cost
            + control_cost
            + angular_velocity_cost
        )
        
    def terminal_cost(self, x):

        nq = self.model.nq
        nv = self.model.nv

        self.cost_data.qpos[:] = x[:nq]
        self.cost_data.qvel[:] = x[nq:nq + nv]

        mujoco.mj_forward(
            self.model,
            self.cost_data
        )

        mouth_pos = self.cost_data.geom_xpos[
            self.mouth_id
        ].copy()

        error = mouth_pos - self.goal_pos

        # Free-body translational velocity
        vel = self.cost_data.qvel[:3]

        # Free-body angular velocity
        omega = self.cost_data.qvel[3:6]

        terminal_position_cost = (
            0.5 * error.T @ self.Qf @ error
        )

        terminal_velocity_cost = (
            0.5 * vel.T @ self.Vf @ vel
        )

        terminal_angular_cost = (
            0.5 * omega.T @ self.Wf @ omega
        )

        return (
            terminal_position_cost
            + terminal_velocity_cost
            + terminal_angular_cost
        )


class PandaReachModel:
    """
    7-DoF torque-controlled Franka Panda reaching model.

    State:
        x = [q; qdot] in R^14

    Control:
        u = joint torques in R^7

    MuJoCo itself does NOT clip torque commands in panda_torque.xml.
    Use u_min/u_max inside Box-iLQR.
    """

    def __init__(
        self,
        xml_path="Models/Panda/panda_box_scene.xml",
        horizon=100,
        n_substeps=10,
    ):
        self.xml_path = str(Path(xml_path))

        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        # Separate data object so cost evaluation does not overwrite
        # the dynamics state.
        self.cost_data = mujoco.MjData(self.model)

        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nx = self.nq + self.nv
        self.nu = self.model.nu

        if (self.nq, self.nv, self.nu) != (7, 7, 7):
            raise RuntimeError(
                "Expected torque Panda dimensions nq=7, nv=7, nu=7, "
                f"but got nq={self.nq}, nv={self.nv}, nu={self.nu}. "
                "Run setup_panda_model.py first."
            )

        self.horizon = int(horizon)
        self.n_substeps = int(n_substeps)

        # panda_torque.xml uses a 0.002 s physics timestep.
        self.dt_physics = float(self.model.opt.timestep)
        self.dt = self.dt_physics * self.n_substeps

        # --------------------------------------------------------------
        # Physical torque limits used by Box-iLQR.
        # The MuJoCo plant itself is intentionally left unclipped.
        # --------------------------------------------------------------
        self.u_max = np.array(
            [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0],
            dtype=float,
        )
        self.u_min = -self.u_max

        # --------------------------------------------------------------
        # Initial configuration: based on the Menagerie home pose.
        # --------------------------------------------------------------
        self.q0 = np.array(
            [
                0.0,
                0.0,
                0.0,
                -1.57079,
                0.0,
                1.57079,
                -0.7853,
            ],
            dtype=float,
        )
        self.qd0 = np.zeros(7)

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.q0
        self.data.qvel[:] = self.qd0
        mujoco.mj_forward(self.model, self.data)

        self.x0 = np.concatenate(
            (self.data.qpos.copy(), self.data.qvel.copy())
        )

        # --------------------------------------------------------------
        # IDs.
        # --------------------------------------------------------------
        self.ee_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            "ee_site",
        )
        self.target_geom_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "target",
        )

        if self.ee_site_id < 0:
            raise RuntimeError("ee_site not found.")
        if self.target_geom_id < 0:
            raise RuntimeError("target geom not found.")

        # Target is fixed in world coordinates.
        self.goal_pos = self.model.geom_pos[
            self.target_geom_id
        ].copy()

        # --------------------------------------------------------------
        # Cost.
        #
        # Position is the primary task.
        # Velocity penalty encourages arrival at rest.
        #
        # Control penalty is normalized by torque authority so the
        # 12 Nm wrist joints and 87 Nm proximal joints are comparable.
        # --------------------------------------------------------------
        self.Q_pos = 100.0 * np.eye(3)
        self.Q_vel = 0.05 * np.eye(7)

        control_weight = 0.10
        self.R = control_weight * np.diag(
            1.0 / (self.u_max ** 2)
        )

        self.Qf_pos = 5000.0 * np.eye(3)
        self.Qf_vel = 10.0 * np.eye(7)

    # ------------------------------------------------------------------
    # State / kinematics helpers
    # ------------------------------------------------------------------

    def set_state(self, data, x):
        x = np.asarray(x, dtype=float)

        if x.shape != (self.nx,):
            raise ValueError(
                f"x must have shape ({self.nx},), got {x.shape}"
            )

        data.qpos[:] = x[: self.nq]
        data.qvel[:] = x[self.nq :]
        mujoco.mj_forward(self.model, data)

    def end_effector_position(self, x):
        self.set_state(self.cost_data, x)
        return self.cost_data.site_xpos[self.ee_site_id].copy()

    def position_error(self, x):
        return self.end_effector_position(x) - self.goal_pos

    # ------------------------------------------------------------------
    # Discrete dynamics
    # ------------------------------------------------------------------

    def propagate_dynamics(self, x, u, t=None):
        """
        One control interval:
            x_{k+1} = f(x_k, u_k)

        u is held constant for n_substeps MuJoCo physics steps.

        IMPORTANT:
        No clipping is performed here. Standard iLQR can be unconstrained;
        Box-iLQR should impose self.u_min <= u <= self.u_max itself.
        """
        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float)

        if u.shape != (self.nu,):
            raise ValueError(
                f"u must have shape ({self.nu},), got {u.shape}"
            )

        self.data.qpos[:] = x[: self.nq]
        self.data.qvel[:] = x[self.nq :]
        self.data.ctrl[:] = u

        if t is not None:
            # If t is an integer optimizer index:
            self.data.time = float(t) * self.dt

        mujoco.mj_forward(self.model, self.data)

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

            if (
                not np.all(np.isfinite(self.data.qpos))
                or not np.all(np.isfinite(self.data.qvel))
                or not np.all(np.isfinite(self.data.qacc))
            ):
                raise FloatingPointError(
                    "MuJoCo became unstable during propagation."
                )

        return np.concatenate(
            (
                self.data.qpos.copy(),
                self.data.qvel.copy(),
            )
        )

    # ------------------------------------------------------------------
    # Costs
    # ------------------------------------------------------------------

    def cost(self, x, u, sigma=100):

        self.set_state(self.cost_data, x)

        p = self.cost_data.site_xpos[self.ee_site_id]
        e = p - self.goal_pos

        qvel = self.cost_data.qvel

        u = np.asarray(u, dtype=float).reshape(-1)

#        lower_slack = u - self.u_min
#        upper_slack = self.u_max - u

#         Log barrier is defined only strictly inside the bounds
#        if np.any(lower_slack <= 0) or np.any(upper_slack <= 0):
#            return np.inf

#        log_barrier_cost = -sigma * np.sum(
#            np.log(lower_slack)
#            + np.log(upper_slack)
#        )

        return (
            0.5 * e.T @ self.Q_pos @ e
            + 0.5 * qvel.T @ self.Q_vel @ qvel
            + 0.5 * u.T @ self.R @ u
        )

    def terminal_cost(self, x):
        """
        Terminal cost:
          1/2 e_p' Qf_pos e_p
        + 1/2 qdot' Qf_vel qdot
        """
        self.set_state(self.cost_data, x)

        p = self.cost_data.site_xpos[self.ee_site_id]
        e = p - self.goal_pos
        qvel = self.cost_data.qvel

        return (
            0.5 * e.T @ self.Qf_pos @ e
            + 0.5 * qvel.T @ self.Qf_vel @ qvel
        )

    def trajectory_cost(self, x_traj, u_traj):
        J = 0.0

        for k in range(self.horizon):
            J += self.cost(
                x_traj[:, k],
                u_traj[:, k],
            )

        J += self.terminal_cost(x_traj[:, -1])

        return J

    # ------------------------------------------------------------------
    # Gauss-Newton quadratization for iLQR
    # ------------------------------------------------------------------

    def quadratize_cost(self, x, u, terminal=False):
        """
        Returns:
            c_x   : (14,)
            c_u   : (7,)
            c_xx  : (14,14)
            c_uu  : (7,7)
            c_ux  : (7,14)

        For Cartesian position, uses Gauss-Newton:
            Hessian approx Jp.T @ Q @ Jp

        This avoids injecting unnecessary indefinite curvature into
        the iLQR backward pass.
        """
        self.set_state(self.cost_data, x)

        p = self.cost_data.site_xpos[self.ee_site_id]
        e = p - self.goal_pos

        jacp = np.zeros((3, self.nv))
        jacr = np.zeros((3, self.nv))

        mujoco.mj_jacSite(
            self.model,
            self.cost_data,
            jacp,
            jacr,
            self.ee_site_id,
        )

        if terminal:
            Qp = self.Qf_pos
            Qv = self.Qf_vel
        else:
            Qp = self.Q_pos
            Qv = self.Q_vel

        c_x = np.zeros(self.nx)
        c_xx = np.zeros((self.nx, self.nx))

        # Position part
        c_x[: self.nq] = jacp.T @ Qp @ e
        c_xx[: self.nq, : self.nq] = jacp.T @ Qp @ jacp

        # Velocity part
        qvel = self.cost_data.qvel
        c_x[self.nq :] = Qv @ qvel
        c_xx[self.nq :, self.nq :] = Qv

        if terminal:
            c_u = np.zeros(self.nu)
            c_uu = np.zeros((self.nu, self.nu))
            c_ux = np.zeros((self.nu, self.nx))
        else:
            c_u = self.R @ u
            c_uu = self.R.copy()
            c_ux = np.zeros((self.nu, self.nx))

        # Force exact numerical symmetry.
        c_xx = 0.5 * (c_xx + c_xx.T)
        c_uu = 0.5 * (c_uu + c_uu.T)

        return c_x, c_u, c_xx, c_uu, c_ux

    # ------------------------------------------------------------------
    # Initial control guess
    # ------------------------------------------------------------------

    def gravity_compensation(self, x=None):
        """
        At qdot=0, qfrc_bias is primarily gravity compensation.
        Returns a useful torque initial guess for iLQR.
        """
        if x is None:
            x = self.x0

        self.set_state(self.cost_data, x)

        # qfrc_bias contains Coriolis/centrifugal/gravity terms.
        # With qdot = 0 at x0, this is a good holding torque.
        return self.cost_data.qfrc_bias[:7].copy()

    def initial_control_guess(self):
        u_hold = self.gravity_compensation(self.x0)
        return np.repeat(
            u_hold[:, None],
            self.horizon,
            axis=1,
        )

    def rollout(self, u_traj, x0=None):
        if x0 is None:
            x0 = self.x0

        x_traj = np.zeros((self.nx, self.horizon + 1))
        x_traj[:, 0] = x0

        for k in range(self.horizon):
            x_traj[:, k + 1] = self.propagate_dynamics(
                x_traj[:, k],
                u_traj[:, k],
                k,
            )

        return x_traj
    
    def check_constraint_violation(self, x, u):
        
        invalid_mask = (u <= self.u_min) | (u >= self.u_max)
        failed_indices = np.where(invalid_mask)[0]
        
        all_pass = np.all((u > self.u_min) & (u < self.u_max))
        
        return all_pass, failed_indices
        
