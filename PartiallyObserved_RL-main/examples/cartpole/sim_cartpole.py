"""
CartPole Dynamics for iLQR - Forward Euler Integration
---------------------------
Dynamics for inverted pendulum on a cart system.
State: [x, x_dot, theta, theta_dot]
Control: [force]

The cart-pole system is described by:
    (M + m)*x_ddot + m*l*theta_ddot*cos(theta) - m*l*theta_dot^2*sin(theta) = force
    l*theta_ddot + x_ddot*cos(theta) - g*sin(theta) = 0

where:
    x: cart position
    theta: pole angle from vertical (theta=0 is upright, theta=pi is downward)
    M: cart mass
    m: pole mass
    l: pole length
    g: gravity
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class SimulateCartPole:
    def __init__(self, nx, nu, dt):
        """
        Initialize CartPole simulator
        
        Args:
            nx: state dimension (4)
            nu: control dimension (1)
            dt: timestep
        """
        self.nx = nx
        self.nu = nu
        self.dt = dt
        
        # Physical parameters
        self.g = 9.81      # gravity (m/s^2)
        self.M = 1.0       # cart mass (kg)
        self.m = 0.01      # pole mass (kg)
        self.l = 0.6       # pole length (m)
    
    def _angleNormalize(self, theta):
        """Normalize angle to [-pi, pi]"""
        return theta  # No normalization for iLQR compatibility
    
    def cartpole_dynamics(self, state, force):
        """
        Return the derivative vector for the CartPole equations.
        
        Args:
            state: [x, x_dot, theta, theta_dot]
            force: control force applied to cart
            
        Returns:
            derivatives: [x_dot, x_ddot, theta_dot, theta_ddot]
        """
        x = state[0]
        x_dot = state[1]
        theta = state[2]
        theta_dot = state[3]
        
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        
        # Compute intermediate term
        term1 = self.m / (self.M + self.m - self.m * (costheta**2))
        term2 = term1 * (self.g * sintheta * costheta - self.l * sintheta * (theta_dot**2))
        
        # State derivatives without control
        dx = x_dot
        dx_dot = term2
        dtheta = theta_dot
        dtheta_dot = (self.g * sintheta / self.l) + (costheta / self.l) * term2
        
        # Add control influence
        control_term = term1 * force / self.m
        dx_dot += control_term
        dtheta_dot += (costheta / self.l) * control_term
        
        return np.array([dx, dx_dot, dtheta, dtheta_dot])
    
    def onestep_euler(self, state_init, n_per_step, u=0.0):
        """
        One step Forward Euler integration with n_per_step substeps
        
        Args:
            state_init: initial state [4]
            n_per_step: number of integration substeps
            u: control input (scalar)
            
        Returns:
            states: trajectory over substeps [n_per_step x 4]
        """
        h = self.dt / n_per_step
        state = state_init.copy()
        states = np.zeros((n_per_step, self.nx))
        
        for i in range(n_per_step):
            # Forward Euler: x(k+1) = x(k) + h * f(x(k), u)
            derivatives = self.cartpole_dynamics(state, u)
            state = state + h * derivatives
            
            # Normalize angle (optional - can be removed for iLQR)
            state[2] = self._angleNormalize(state[2])
            
            states[i, :] = state
        
        return states
    
    def simulate(self, x_init, u_scalar):
        """
        Single step simulation (for iLQR compatibility) using Forward Euler
        
        Args:
            x_init: Initial state [4] (can be flattened array)
            u_scalar: Control input (scalar or [1] array)
        
        Returns:
            next_state: Next state [4]
        """
        x_init = np.array(x_init).flatten()
        u_scalar = float(np.array(u_scalar).flatten()[0]) if hasattr(u_scalar, '__iter__') else float(u_scalar)
        
        # Single Forward Euler step
        derivatives = self.cartpole_dynamics(x_init, u_scalar)
        next_state = x_init + self.dt * derivatives
        next_state[2] = self._angleNormalize(next_state[2])
        
        return next_state
    
    def simulate_trajectory(self, state_init=np.array([0.0, 0.0, np.pi, 0.0]), 
                          u=np.array([0.0]), horizon=1, n_per_step=20):
        """
        Simulate CartPole trajectory using Forward Euler
        
        Args:
            state_init: Initial state [x, x_dot, theta, theta_dot]
            u: Control sequence [horizon] or [horizon x 1]
            horizon: Number of timesteps
            n_per_step: Integration substeps per timestep
        
        Returns:
            Y: State trajectory [total_steps x 4]
        """
        if u.shape[0] != horizon:
            u = np.zeros(horizon)
        u = u.flatten()
        
        total_steps = (horizon + 1)
        self.T = np.linspace(0, (horizon + 1) * self.dt, total_steps)
        self.Y = np.zeros((total_steps, self.nx))
        self.U = u

        # Store initial state
        self.Y[0, :] = state_init
        
        state = state_init.copy()
        
        for i in range(horizon):
            states = self.onestep_euler(state, n_per_step, u[i])
            self.Y[i+1, :] = states[-1]
            state = states[-1]
        
        return self.Y
    
    def draw_figure(self, save_to_path=None):
        """Plot trajectory results"""
        fig = plt.figure(figsize=(14, 10))
        
        # Cart position
        plt.subplot(3, 2, 1)
        plt.plot(self.T, self.Y[:, 0], '-ob', markersize=3)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        plt.title('Cart Position (x)')
        plt.grid(True, alpha=0.3)
        
        # Cart velocity
        plt.subplot(3, 2, 2)
        plt.plot(self.T, self.Y[:, 1], '-og', markersize=3)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (m/s)')
        plt.title('Cart Velocity (ẋ)')
        plt.grid(True, alpha=0.3)
        
        # Pole angle
        plt.subplot(3, 2, 3)
        plt.plot(self.T, np.degrees(self.Y[:, 2]), '-or', markersize=3)
        plt.axhline(y=0, color='g', linestyle='--', alpha=0.5, label='Upright')
        plt.axhline(y=180, color='b', linestyle='--', alpha=0.5, label='Downward')
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (deg)')
        plt.title('Pole Angle (θ)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Pole angular velocity
        plt.subplot(3, 2, 4)
        plt.plot(self.T, self.Y[:, 3], '-om', markersize=3)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Angular velocity (rad/s)')
        plt.title('Pole Angular Velocity (θ̇)')
        plt.grid(True, alpha=0.3)
        
        # Phase portrait: theta vs theta_dot
        plt.subplot(3, 2, 5)
        plt.plot(np.degrees(self.Y[:, 2]), self.Y[:, 3], '-ok', markersize=2)
        plt.scatter(np.degrees(self.Y[0, 2]), self.Y[0, 3], c='g', s=100, 
                   marker='o', label='Start', zorder=5)
        plt.scatter(np.degrees(self.Y[-1, 2]), self.Y[-1, 3], c='r', s=100, 
                   marker='x', label='End', zorder=5)
        plt.xlabel('Angle (deg)')
        plt.ylabel('Angular velocity (rad/s)')
        plt.title('Phase Portrait')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Control inputs
        plt.subplot(3, 2, 6)
        t_control = np.linspace(0, len(self.U) * self.dt, len(self.U))
        plt.step(t_control, self.U, '-b', where='post', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Force (N)')
        plt.title('Control Input')
        plt.grid(True, alpha=0.3)
        
        plt.suptitle('CartPole Swing-Up Trajectory (Forward Euler)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_to_path is not None:
            plt.savefig(save_to_path, format='png', dpi=150)
            print(f"Figure saved to: {save_to_path}")
        
        plt.show()

if __name__ == '__main__':
    # Setup paths
    cwd = Path.cwd()
    file_loc = cwd / "examples/cartpole"
    file_loc.mkdir(parents=True, exist_ok=True)
    
    save_path = file_loc / "cartpole_trajectory_euler.png"
    
    # CartPole parameters
    nx, nu, dt = 4, 1, 0.01  # Smaller dt recommended for Forward Euler stability
    
    # Initial state: pole hanging down
    state_init = np.array([0.0, 0.0, np.pi, 0.0])
    
    # Time horizon
    time_horizon = 30
    
    # Test control (zero or custom)
    # control = np.zeros(time_horizon)
    control = 10.0 * np.sin(np.linspace(0, 4*np.pi, time_horizon))
    
    print('='*70)
    print('CARTPOLE SWING-UP SIMULATION (FORWARD EULER)')
    print('='*70)
    print(f'Initial state: {state_init}')
    print(f'[x={state_init[0]:.2f}m, ẋ={state_init[1]:.2f}m/s, '
          f'θ={np.degrees(state_init[2]):.1f}°, θ̇={state_init[3]:.2f}rad/s]')
    print(f'Horizon: {time_horizon} steps ({time_horizon*dt:.1f}s)')
    print(f'Integration: Forward Euler with dt={dt}s')
    print('='*70 + '\n')
    
    # Create simulator
    sim = SimulateCartPole(nx, nu, dt)
    
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
    print(f'[x={final_state[0]:.4f}m, ẋ={final_state[1]:.4f}m/s, '
          f'θ={np.degrees(final_state[2]):.2f}°, θ̇={final_state[3]:.4f}rad/s]')
    
    theta_error = np.abs(final_state[2])
    success = theta_error < np.deg2rad(10)
    print(f'\nAngle error: {np.degrees(theta_error):.2f}°')
    print(f'Success: {"YES ✓" if success else "NO ✗"}')
    print('='*70 + '\n')
    
    # Draw figure
    sim.draw_figure(save_to_path=save_path)