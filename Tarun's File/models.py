import numpy as np
from scipy.integrate import solve_ivp
import mujoco
import mujoco.viewer
import imageio

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
            
class Pendulum2():
    def __init__(self, int_type = 'euler', steps = 1000, u_max = 1, u_min = -1, \
                 x0 = np.array([0, 0]), \
                 xg = np.array([np.pi, 0]), dt = 0.02, horizon = 500):
        self.nx = 2                                     #no. of states
        self.nu = 1                                     #no. of control inputs
        
        self.name = 'pendulum'
        self.mujoco_model = mujoco.MjModel.from_xml_path('D:/PythonProjects/EDP/iLQR/Python Code/Models/pendulum.xml')
        #self.mujoco_model.opt.timestep = dt
        self.data = mujoco.MjData(self.mujoco_model)
        #self.data.qpos[:] = x0[:self.nx//2]
        #self.data.qvel[:] = x0[self.nx//2:]
        mujoco.mj_forward(self.mujoco_model, self.data)
        self.v = mujoco.viewer.launch_passive(self.mujoco_model, self.data)
        
        self.dt = dt
        self.dt_ratio = int(self.dt/self.mujoco_model.opt.timestep)
        self.m = 0.5
        self.L = 0.5
        self.g = 9.81
        
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
        if t == 0:
            mujoco.mj_resetData(self.mujoco_model, self.data)
        self.data.qpos[:] = x[:self.nx//2]
        self.data.qvel[:] = x[self.nx//2:]
        for _ in range(self.dt_ratio):
            self.data.ctrl[:] = u
            mujoco.mj_step(self.mujoco_model, self.data)
        next_state = np.hstack((self.data.qpos[:], self.data.qvel[:]))
        #next_state = propagate_forward(self, x, u, t)
        if t%1 == 0:
            self.v.sync()
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
    
    def simulate_controls(self, x, u):
        renderer = mujoco.Renderer(self.mujoco_model, height=480, width=640)
        frames = []
        
        for t in range(self.horizon):
            self.data.ctrl = u[t]
            #self.data.qpos[t] = x[0]
            #self.data.qvel[t] = x[1]
            mujoco.mj_step(self.mujoco_model, self.data)
            renderer.update_scene(self.data)
            pixels = renderer.render()
            frames.append(pixels.copy())
        renderer.close()
        imageio.mimsave("sim_video.mp4", frames, fps = 100)
            #next_state = np.hstack((self.data.qpos[:], self.data.qvel[:]))
            #if t%1 == 0:
            #    self.v.sync()
        return

class Pendulum():
    def __init__(self, int_type = 'euler', steps = 1000, u_max = 1, u_min = -1, \
                 x0 = np.array([0,0]), \
                 xg = np.array([np.pi, 0]), dt = 0.01, horizon = 500):
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
        #if t%10 == 0:
            #print()
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
                 xg = np.array([0, 0, 0, 0]), dt = 0.01, horizon = 1000):
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
        self.Q = 10 * self.dt * np.eye(self.nx)          #incremental state penalty
        self.R = 10  * self.dt * np.eye(self.nu)                                    #Incremental cost penalty
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
        f = np.zeros(self.nx)
        g = np.zeros(self.nx)
        x, x_dot, theta, theta_dot = x
        s_th = np.sin(theta)
        c_th = np.cos(theta)
        
        term1 = self.m/(self.M + self.m - self.m*c_th**2)
        term2 = term1 * (self.g * s_th * c_th - self.L * s_th * (theta_dot**2))
        
        f[0] = x_dot
        f[1] = term2
        f[2] = theta_dot
        f[3] = (self.g * s_th + term2 * c_th) / self.L
        
        g[1] = term1 * u / self.m
        g[3] = (c_th / self.L) * term1 * u / self.m
        
        state_dot = f + g
        
        #u = u.item()
        #x_pos, x_dot, theta, theta_dot = x
        #T1 = self.m/(self.M + self.m - self.m*(np.cos(theta)**2))
        #T2 = T1*(self.g*np.sin(theta)*np.cos(theta) - self.L*np.sin(theta)*theta_dot**2)
        #x_double_dot = T2 + T1*u/self.m
        #theta_double_dot = (self.g * np.sin(theta) / self.L) + (np.cos(theta)/self.L)*(T2+ T1*u/self.m)
        return state_dot
    
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
            cost -= sigma * np.log(u - self.u_min, eps)
            cost -= sigma * np.log(self.u_max - u, eps)

        return cost

    def terminal_cost(self, x):
        state_error = x - self.xg
        terminal_cost = 0.5 * state_error.T @ self.Qf @ state_error
        
        return terminal_cost
            
class Acrobot():
    def __init__(self, int_type = 'euler', steps = 10, u_max = 5, u_min = -5, \
                 x0 = np.array([0,0,0,0]), \
                 xg = np.array([np.pi, 0, 0, 0]), dt = 0.01, horizon = 1000):
        self.name = 'acrobot'
        self.l1 = 1
        self.l2 = 1
        self.m1 = 1
        self.m2 = 1
        self.g = 9.81
        
        #self.lc1 = 0.5
        #self.I1 = 0.5*self.m1*(self.l1**2)/12
        #self.l2 = 1
        #self.lc2 = 0.5
        #self.I2 = 0.5*self.m2*(self.l2**2)/12
        
        self.nx = 4                                     #no. of states
        self.nu = 1                                     #no. of control inputs
        self.dt = dt
        self.horizon = horizon                              #time-step
        self.int_type = int_type                        #type of solver used to integrate dynamics
        self.steps = steps                                #no. of steps of the integrator
        self.Q = 500 * self.dt *np.eye(self.nx)           #incremental state penalty
        self.R = 10 * self.dt                                      #Incremental cost penalty
        self.Qf = 5e4*np.eye(self.nx)                   #Quadratic penalty fro terminal cost
        self.u_max = u_max                              #Maximum value of control
        self.u_min = u_min                              #Minimum value of control
        self.x0 = x0                                    #Initial State
        self.xg = xg                                    #Goal State

    def eom(self, x, u, t):
       # --- Model Parameters ---
        l1 = self.l1
        m1 = self.m1
        I1 = m1*(l1**2)/12
        
        l2 = self.l2
        m2 = self.m2
        I2 = m2*(l2**2)/12
        
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
        M = np.array([[I1 + I2 + m1*(l1**2)/4 + m2*l1*l2*c2 + m2*(l2**2)/4 + m2*(l1**2), I2 + m2*(l2**2)/4 + m2*l1*l2*c2/2], 
                      [I2 + m2*(l2**2)/4 + m2*l1*l2*c2/2, I2 + m2*(l2**2)/4]])

        # C(q, q_dot) - Coriolis and Centrifugal Matrix
        C = np.array([[-m2*l1*l2*th1_dot*th2_dot*s2-m2*l1*l2*(th2_dot**2)*s2/2], 
                      [m2*l1*l2*(th1_dot**2)*s2/2]])

        # G(q) - Gravity Vector
        G = np.array([[m1*g*l1*s1/2 + m2*g*l1*s1 + m2*g*l2*s12/2], 
                      [m2*g*l2*s12/2]])
        
        # B - Input Matrix
        B = np.array([[0],[1]])
        
        # --- Solve for Accelerations (q_ddot) ---
        
        # RHS of the equation: M*q_ddot = (B*U - C - G)
        rhs = B * u - C - G
        
        # Solve the linear system M * q_ddot = rhs for q_ddot
        q_ddot = np.linalg.pinv(M)@rhs
        
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
        if cost_type != "quadratic":
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
    
    
#%% MuJoCo Models

class Walker():
    def __init__(self, int_type = 'euler', steps = 1000, u_max = 1, u_min = -1, \
                 x0 = np.zeros(18), \
                 xg = np.zeros(18), dt = 0.01, horizon = 500):
        
        self.nx = 18                                     #no. of states
        self.nu = 6                                     #no. of control inputs
        self.dt = dt
        self.name = 'walker'
        self.mujoco_model = mujoco.MjModel.from_xml_path('D:/PythonProjects/EDP/iLQR/Python Code/Models/Walker.xml')
        #self.mujoco_model.opt.timestep = dt
        self.data = mujoco.MjData(self.mujoco_model)
        self.data.qpos[:] = x0[:self.nx//2]
        self.data.qvel[:] = x0[self.nx//2:]
        mujoco.mj_forward(self.mujoco_model, self.data)
        self.v = mujoco.viewer.launch_passive(self.mujoco_model, self.data)
        
        self.dt_ratio = int(self.dt/self.mujoco_model.opt.timestep)
        self.m = 0.5
        self.L = 0.5
        self.g = 9.81
        
        self.horizon = horizon                              #time-step
        self.int_type = int_type                        #type of solver used to integrate dynamics
        self.steps = steps                                #no. of steps of the integrator
        self.Q = 3*np.diag(np.ones(self.nx))          #incremental state penalty
        self.R = 3*np.diag(np.ones(self.nu))          #Incremental cost penalty
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
        if t == 0:
            mujoco.mj_resetData(self.mujoco_model, self.data)
        self.data.qpos[:] = x[:self.nx//2]
        self.data.qvel[:] = x[self.nx//2:]
        for _ in range(self.dt_ratio):
            self.data.ctrl[:] = u
            mujoco.mj_step(self.mujoco_model, self.data)
        next_state = np.hstack((self.data.qpos[:], self.data.qvel[:]))
        #next_state = propagate_forward(self, x, u, t)
        if t % 1 == 0:
            self.v.sync()
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
        
        
        quadratic_control_cost = 0.5 * u.T @ self.R @ u * self.dt

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
    
    def simulate_controls(self, x, u):
        renderer = mujoco.Renderer(self.mujoco_model, height=480, width=640)
        frames = []
        
        for t in range(self.horizon):
            self.data.ctrl = u[t]
            #self.data.qpos[t] = x[0]
            #self.data.qvel[t] = x[1]
            mujoco.mj_step(self.mujoco_model, self.data)
            renderer.update_scene(self.data)
            pixels = renderer.render()
            frames.append(pixels.copy())
        renderer.close()
        imageio.mimsave("sim_video.mp4", frames, fps = 1)
            #next_state = np.hstack((self.data.qpos[:], self.data.qvel[:]))
            #if t%1 == 0:
            #    self.v.sync()
        return
    
class Hopper():
    def __init__(self, int_type = 'euler', steps = 800, u_max = 1, u_min = -1, \
                 x0 = np.array([0, 0, -6.03631652e-02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), \
                 xg = np.array([0, 0, -6.03631652e-02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), dt = 0.01, horizon = 100):
        
        self.nx = 7*2                                     #no. of states
        self.nx_alt = 6
        self.nu = 4                                     #no. of control inputs
        self.dt = dt
        self.name = 'hopper'
        self.mujoco_model = mujoco.MjModel.from_xml_path('D:/PythonProjects/EDP/iLQR/Python Code/Models/hopper2.xml')
        #self.mujoco_model.opt.timestep = dt
        self.data = mujoco.MjData(self.mujoco_model)
        self.data.qpos[:] = x0[:self.nx//2]
        self.data.qvel[:] = x0[self.nx//2:]
        mujoco.mj_forward(self.mujoco_model, self.data)
        self.v = mujoco.viewer.launch_passive(self.mujoco_model, self.data)
        
        
        foot_id = mujoco.mj_name2id(self.mujoco_model, mujoco.mjtObj.mjOBJ_GEOM, "foot")
        foot_z = self.data.geom_xpos[foot_id][2]
    
        radius = self.mujoco_model.geom_size[foot_id][0]
        foot_bottom = foot_z - radius
        #print(self.data.geom_xmat[foot_id].reshape(3, 3))
        self.dt_ratio = int(self.dt/self.mujoco_model.opt.timestep)
        self.m = 0.5
        self.L = 0.5
        self.g = 9.81
        
        self.horizon = horizon                              #time-step
        self.int_type = int_type                        #type of solver used to integrate dynamics
        self.steps = steps                                #no. of steps of the integrator
        self.Q = np.diag(np.ones(4))                    #incremental state penalty
        self.R = 0.1*np.diag(np.ones(self.nu))          #Incremental cost penalty
        self.Qf = 200*np.diag(np.ones(4))#200*np.diag(np.ones(2))                   #Quadratic penalty fro terminal cost
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
        mujoco.mj_resetData(self.mujoco_model, self.data)
        self.data.qpos[:] = x[:self.nx//2]
        self.data.qvel[:] = x[self.nx//2:]
        mujoco.mj_forward(self.mujoco_model, self.data)
        for _ in range(self.dt_ratio):
            self.data.ctrl[:] = u
            mujoco.mj_step(self.mujoco_model, self.data)
        next_state = np.hstack((self.data.qpos[:], self.data.qvel[:]))
        #next_state = propagate_forward(self, x, u, t)
        if t % 5 == 0:
            self.v.sync()
        
        toe_posx = self.data.site_xpos[0, 0]
        heel_posx = self.data.site_xpos[1, 0]
        body_vel = self.data.cvel[1, :]
        body_com_xpos = self.data.subtree_com[0, 0]
        #body_com_ypos = self.data.subtree_com[0, 1]
        body_com_zpos = self.data.subtree_com[0, 2]
        
        next_alt_state = np.hstack((next_state[:3], body_com_xpos, toe_posx, heel_posx))
        #alt_x = np.hstack((x))
        
        return next_state, next_alt_state
        
    def cost(self, x, x_alt, u, cost_type="quadratic", sigma=100):
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
        #state_error = x - self.xg
        #body_com_xpos = self.data.subtree_com[0, 0]
        #foot_quaternion = self.data.xquat[5] # Foot Id number is 6
        
        #toe_pos = self.data.site_xpos[0, :]
        #heel_pos = self.data.site_xpos[1, :]
        #body_vel = self.data.cvel[1, :]
        #body_com_xpos = self.data.subtree_com[0, 0]
        #body_com_zpos = self.data.subtree_com[0, 1]
        #x_alt = np.hstack((x[:3], body_com_xpos))#, toe_pos, heel_pos))
        toe_posx = x_alt[4]
        heel_posx = x_alt[5]
        xg_alt = 0.5*(toe_posx-heel_posx)
        aug_state_error = np.hstack((x_alt[:4])) - np.hstack((self.xg[:3], xg_alt))#np.hstack((self.xg[:3], self.xg[7:], np.array([0.5*(toe_pos[0]-heel_pos[0]), 0.17, 0, 0.03963683, -0.08, 0, 0.03963683])))
        #_, alt_state_error = self.alternate_state(x)
        #0.70353782
        #print(self.data.subtree_com[0, :] - np.array([0.0073533,  0,         ]))
        #print(self.data.cvel[0, :])
        quadratic_state_cost = 0.5 * aug_state_error.T @ self.Q @ aug_state_error * self.dt
        
        quadratic_control_cost = 0.5 * u.T @ self.R @ u * self.dt

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
    '''
    def alternate_state(self, x):
        toe_pos = self.data.site_xpos[0, :]
        heel_pos = self.data.site_xpos[1, :]
        body_vel = self.data.cvel[1, :]
        body_com_xpos = self.data.subtree_com[0, 0]
        body_com_zpos = self.data.subtree_com[0, 2]
        #alt_x = np.hstack((body_com_xpos, body_com_zpos))
        #alt_state_error = alt_x - np.hstack((0.5*(toe_pos[0]-heel_pos[0]), 0.7))
        alt_x = np.hstack((x))
        self.xg[0] = 0.5*(toe_pos[0]-heel_pos[0])
        alt_state_error = alt_x - self.xg
        return alt_x, alt_state_error
    '''
    def terminal_cost(self, x, x_alt):
        #state_error = x - self.xg
        
        toe_posx = x_alt[4]
        heel_posx = x_alt[5]
        xg_alt = 0.5*(toe_posx-heel_posx)
        #alt_x = np.hstack((x[:3], body_com_xpos))
        aug_state_error = np.hstack((x_alt[:4])) - np.hstack((self.xg[:3], xg_alt))
        #alt_state_error = alt_x - np.hstack((self.xg, 0.5*(toe_pos[0]-heel_pos[0])))#np.hstack((self.xg[:3],  self.xg[7:], np.array([0.5*(toe_pos[0]-heel_pos[0]), 0.17, 0, 0.03963683, -0.08, 0, 0.03963683])))
        #_, alt_state_error = self.alternate_state(x)
        
        terminal_cost = 0.5 * aug_state_error.T @ self.Qf @ aug_state_error
        return terminal_cost
    
    def simulate_controls(self, x, u):
        renderer = mujoco.Renderer(self.mujoco_model, height=480, width=640)
        frames = []
        
        for t in range(self.horizon):
            self.data.ctrl = u[:, t]
            #self.data.qpos[t] = x[0]
            #self.data.qvel[t] = x[1]
            mujoco.mj_step(self.mujoco_model, self.data)
            renderer.update_scene(self.data)
            pixels = renderer.render()
            frames.append(pixels.copy())
        renderer.close()
        imageio.mimsave("sim_video.mp4", frames, fps = 10)
            #next_state = np.hstack((self.data.qpos[:], self.data.qvel[:]))
            #if t%1 == 0:
            #    self.v.sync()
        return