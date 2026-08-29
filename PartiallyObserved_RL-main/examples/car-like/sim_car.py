"""
Car Dynamics for iLQR
---------------------------
Dynamics for car-like robot using bicycle/kinematic model.
State: [x, y, theta, v]
Control: [a, delta]

The car model is described by:
    x_dot = v * cos(theta)
    y_dot = v * sin(theta)
    theta_dot = v * tan(delta) / L
    v_dot = a

where:
    x, y: position
    theta: heading angle
    v: velocity
    a: acceleration
    delta: steering angle
    L: wheelbase
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class SimulateCar:
    def __init__(self, nx, nu, dt):
        """
        Initialize Car simulator
        
        Args:
            nx: state dimension (4)
            nu: control dimension (2)
            dt: timestep
        """
        self.nx = nx
        self.nu = nu
        self.dt = dt
        
        # Physical parameters
        self.L = 0.58  # wheelbase (m)
        self.u_max = 7.0  # max control magnitude
    
    def _angleNormalize(self, theta):
        """Normalize angle to [-pi, pi]"""
        return theta  # No normalization for iLQR compatibility
    
    def car_dynamics(self, state, control):
        """
        Return the derivative vector for the car equations.
        
        Args:
            state: [x, y, theta, v]
            control: [a, delta] - acceleration and steering angle
            
        Returns:
            derivatives: [x_dot, y_dot, theta_dot, v_dot]
        """
        x = state[0]
        y = state[1]
        theta = state[2]
        v = state[3]
        
        a = control[0]      # acceleration
        delta = control[1]  # steering angle
        
        # Bicycle/Kinematic model
        x_dot = v * np.cos(theta)
        y_dot = v * np.sin(theta)
        theta_dot = v * np.tan(delta) / self.L
        v_dot = a
        
        return np.array([x_dot, y_dot, theta_dot, v_dot])
    
    def onestep_rk4(self, state_init, n_per_step, u=np.array([0.0, 0.0])):
        """
        One step RK4 integration with n_per_step substeps
        
        Args:
            state_init: initial state [4]
            n_per_step: number of integration substeps
            u: control input [2] - [acceleration, steering]
            
        Returns:
            states: trajectory over substeps [n_per_step x 4]
        """
        h = self.dt / n_per_step
        state = state_init.copy()
        states = np.zeros((n_per_step, self.nx))
        
        for i in range(n_per_step):
            # RK4 integration
            k1 = self.car_dynamics(state, u)
            k2 = self.car_dynamics(state + 0.5 * h * k1, u)
            k3 = self.car_dynamics(state + 0.5 * h * k2, u)
            k4 = self.car_dynamics(state + h * k3, u)
            
            # Update state
            state = state + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
            # Normalize angle (optional)
            state[2] = self._angleNormalize(state[2])
            
            states[i, :] = state
        
        return states
    
    def simulate_car(self, x_init, u_input):
        """
        Single step simulation (for iLQR compatibility)
        
        Args:
            x_init: Initial state [4] (can be flattened array)
            u_input: Control input [2] - [acceleration, steering]
        
        Returns:
            next_state: Next state [4]
        """
        x_init = np.array(x_init).flatten()
        u_input = np.array(u_input).flatten()
        
        # Single RK4 step
        k1 = self.car_dynamics(x_init, u_input)
        k2 = self.car_dynamics(x_init + 0.5 * self.dt * k1, u_input)
        k3 = self.car_dynamics(x_init + 0.5 * self.dt * k2, u_input)
        k4 = self.car_dynamics(x_init + self.dt * k3, u_input)
        
        next_state = x_init + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        next_state[2] = self._angleNormalize(next_state[2])
        
        return next_state
    
    def simulate_trajectory(self, state_init=np.array([0.0, 0.0, 0.0, 0.0]), 
                     u=None, horizon=1, n_per_step=1):
        """
        Simulate Car trajectory
        
        Args:
            state_init: Initial state [x, y, theta, v]
            u: Control sequence [horizon x 2] or None
            horizon: Number of timesteps
            n_per_step: Integration substeps per timestep
        
        Returns:
            Y: State trajectory [total_steps x 4]
        """
        if u is None:
            u = np.zeros((horizon, 2))
        
        if u.shape[0] != horizon:
            u = np.zeros((horizon, 2))

        total_steps = (horizon + 1)
        self.T = np.linspace(0, (horizon + 1) * self.dt, total_steps)
        self.Y = np.zeros((total_steps, self.nx))
        self.U = u

        # Store initial state
        self.Y[0, :] = state_init
                
        state = state_init.copy()
        
        for i in range(horizon):
            states = self.onestep_rk4(state, n_per_step, u[i])
            self.Y[i+1, :] = states[-1]
            state = states[-1]
        
        return self.Y
    
    def draw_figure(self, save_to_path=None):
        """Plot trajectory results"""
        fig = plt.figure(figsize=(14, 10))
        
        # X position
        plt.subplot(3, 3, 1)
        plt.plot(self.T, self.Y[:, 0], '-ob', markersize=3)
        plt.xlabel('Time (s)')
        plt.ylabel('X Position (m)')
        plt.title('X Position')
        plt.grid(True, alpha=0.3)
        
        # Y position
        plt.subplot(3, 3, 2)
        plt.plot(self.T, self.Y[:, 1], '-og', markersize=3)
        plt.xlabel('Time (s)')
        plt.ylabel('Y Position (m)')
        plt.title('Y Position')
        plt.grid(True, alpha=0.3)
        
        # Heading angle
        plt.subplot(3, 3, 3)
        plt.plot(self.T, np.degrees(self.Y[:, 2]), '-or', markersize=3)
        plt.xlabel('Time (s)')
        plt.ylabel('Heading (deg)')
        plt.title('Heading Angle (θ)')
        plt.grid(True, alpha=0.3)
        
        # Velocity
        plt.subplot(3, 3, 4)
        plt.plot(self.T, self.Y[:, 3], '-om', markersize=3)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (m/s)')
        plt.title('Velocity (v)')
        plt.grid(True, alpha=0.3)
        
        # 2D trajectory
        plt.subplot(3, 3, 5)
        plt.plot(self.Y[:, 0], self.Y[:, 1], '-b', linewidth=2)
        plt.scatter(self.Y[0, 0], self.Y[0, 1], c='g', s=100, 
                   marker='o', label='Start', zorder=5)
        plt.scatter(self.Y[-1, 0], self.Y[-1, 1], c='r', s=100, 
                   marker='x', label='End', zorder=5)
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title('2D Trajectory')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # Control: Acceleration
        plt.subplot(3, 3, 6)
        t_control = np.linspace(0, len(self.U) * self.dt, len(self.U))
        plt.step(t_control, self.U[:, 0], '-b', where='post', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration (m/s²)')
        plt.title('Control Input: Acceleration')
        plt.grid(True, alpha=0.3)
        
        # Control: Steering angle
        plt.subplot(3, 3, 7)
        plt.step(t_control, (self.U[:, 1]), '-r', where='post', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Steering Angle (rad)')
        plt.title('Control Input: Steering')
        plt.grid(True, alpha=0.3)
        
        # Speed vs heading
        plt.subplot(3, 3, 8)
        plt.plot(np.degrees(self.Y[:, 2]), self.Y[:, 3], '-k', linewidth=2)
        plt.scatter(np.degrees(self.Y[0, 2]), self.Y[0, 3], c='g', s=100, marker='o', label='Start')
        plt.scatter(np.degrees(self.Y[-1, 2]), self.Y[-1, 3], c='r', s=100, marker='x', label='End')
        plt.xlabel('Heading (deg)')
        plt.ylabel('Velocity (m/s)')
        plt.title('Speed vs Heading')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Trajectory with heading arrows
        plt.subplot(3, 3, 9)
        skip = max(1, len(self.Y) // 20)  # Show ~20 arrows
        plt.plot(self.Y[:, 0], self.Y[:, 1], '-b', linewidth=2, alpha=0.5)
        for i in range(0, len(self.Y), skip):
            dx = 0.1 * np.cos(self.Y[i, 2])
            dy = 0.1 * np.sin(self.Y[i, 2])
            plt.arrow(self.Y[i, 0], self.Y[i, 1], dx, dy, 
                     head_width=0.05, head_length=0.05, fc='red', ec='red')
        plt.scatter(self.Y[0, 0], self.Y[0, 1], c='g', s=100, marker='o', label='Start', zorder=5)
        plt.scatter(self.Y[-1, 0], self.Y[-1, 1], c='r', s=100, marker='x', label='End', zorder=5)
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title('Trajectory with Heading')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        plt.suptitle('Car Dynamics Simulation', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_to_path is not None:
            plt.savefig(save_to_path, format='png', dpi=150)
            print(f"Figure saved to: {save_to_path}")
        
        plt.show()


if __name__ == '__main__':
    # Setup paths
    cwd = Path.cwd()
    file_loc = cwd / "examples/car-like"
    file_loc.mkdir(parents=True, exist_ok=True)
    
    save_path = file_loc / "car_trajectory.png"
    
    # Car parameters
    nx, nu, dt = 4, 2, 0.1
    
    # Initial state: [x, y, theta, v]
    state_init = np.array([0.0, 0.0, np.pi/3, 0.0])
    
    # Time horizon
    time_horizon = 30
    
    # Test control: sinusoidal acceleration and steering
    control = np.zeros((time_horizon, 2))
    control[:, 0] = 1.0 * np.sin(np.linspace(0, 2*np.pi, time_horizon))  # acceleration
    control[:, 1] = 0.3 * np.sin(np.linspace(0, 4*np.pi, time_horizon))  # steering angle
    control = np.load('examples/car-like/u_test.npy').T#.reshape((30,2))
    
    
    print('='*70)
    print('CAR DYNAMICS SIMULATION')
    print('='*70)
    print(f'Initial state: {state_init}')
    print(f'[x={state_init[0]:.2f}m, y={state_init[1]:.2f}m, '
          f'θ={np.degrees(state_init[2]):.1f}°, v={state_init[3]:.2f}m/s]')
    print(f'Horizon: {time_horizon} steps ({time_horizon*dt:.1f}s)')
    print(f'Wheelbase: 0.58m')
    print('='*70 + '\n')
    
    # Create simulator
    sim = SimulateCar(nx, nu, dt)
    
    # Run simulation
    trajectory = sim.simulate_trajectory(
        state_init=state_init,
        u=control,
        horizon=time_horizon,
        n_per_step=1
    )
    
    # Print final state
    final_state = trajectory[-1]
    print('\nFinal state:')
    print(f'[x={final_state[0]:.4f}m, y={final_state[1]:.4f}m, '
          f'θ={np.degrees(final_state[2]):.2f}°, v={final_state[3]:.4f}m/s]')
    
    distance_traveled = np.sqrt(final_state[0]**2 + final_state[1]**2)
    print(f'\nDistance traveled: {distance_traveled:.4f}m')
    print('='*70 + '\n')
    
    # Draw figure
    sim.draw_figure(save_to_path=save_path)