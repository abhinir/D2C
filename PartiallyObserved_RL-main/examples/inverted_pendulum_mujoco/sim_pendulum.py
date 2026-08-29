import numpy as np
import mujoco
import matplotlib.pyplot as plt
from pathlib import Path

class SimulatePendulum:
    def __init__(self, nx, nu, dt, model_path="pendulum.xml"):
        """
        Initialize MuJoCo CartPole simulator
        
        Args:
            nx: state dimension (2)
            nu: control dimension (1)
            dt: timestep
            model_path: path to MuJoCo XML file
        """
        self.nx = nx
        self.nu = nu
        self.dt = dt
        
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Set timestep
        self.model.opt.timestep = dt
        
    
    def _angleNormalize(self, theta):
        """Normalize angle to [-pi, pi]"""
        return ((theta + np.pi) % (2 * np.pi)) - np.pi
    
    def reset(self, init_state=np.array([np.pi, 0.0])):
        """
        Reset simulation to initial state
        
        Args:
            init_state: [theta, theta_dot]
        """
        mujoco.mj_resetData(self.model, self.data)
        
        # Set initial state
        # MuJoCo state order: positions then velocities
        self.data.qpos[0] = init_state[0]  # pole angle
        self.data.qvel[0] = init_state[1]  # pole angular velocity
        
        # Forward to update derived quantities
        mujoco.mj_forward(self.model, self.data)
    
    def get_state(self):
        """
        Get current state in format [theta, theta_dot]
        """
        theta = self.data.qpos[0]
        theta_dot = self.data.qvel[0]
        
        # Normalize theta
        theta = self._angleNormalize(theta)
        
        return np.array([theta, theta_dot])
    
    def step(self, u):
        """
        Take one simulation step with control input
        
        Args:
            u: control force (scalar or array)
        """
        # Set control
        self.data.ctrl[0] = u[0]
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        
        return self.get_state()
    
    def simulate_trajectory(self, y_init=np.array([np.pi, 0.0]), 
                          u=np.array([0.0]), horizon=1):
        """
        Simulate CartPole trajectory
        
        Args:
            y_init: Initial state [theta, theta_dot]
            u: Control sequence [horizon x 1]
            horizon: Number of timesteps
        
        Returns:
            Y: State trajectory [horizon+1 x 2]
        """
        if u.shape[0] != horizon:
            u = np.zeros([horizon])
        
        # Reset to initial state
        self.reset(y_init)
        
        # Storage
        self.T = np.arange(horizon + 1) * self.dt
        self.Y = np.zeros((horizon + 1, self.nx))
        self.U = u
        
        # Store initial state
        self.Y[0, :] = self.get_state()
        
        # Simulate forward
        for i in range(horizon):
            state = self.step([u[i]])
            self.Y[i + 1, :] = state
        
        return self.Y
    
    def draw_figure(self, save_to_path=None):
        """Plot trajectory results"""
        fig = plt.figure(figsize=(14, 10))
        
        # Pendulum angle
        plt.subplot(3, 2, 1)
        plt.plot(self.T, self.Y[:, 0], '-ob', markersize=3)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (rad)')
        plt.title('Angle')
        plt.grid(True, alpha=0.3)
        
        # Angular velocity
        plt.subplot(3, 2, 2)
        plt.plot(self.T, self.Y[:, 1], '-og', markersize=3)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Angular velocity (rad/s)')
        plt.title('Angular velocity')
        plt.grid(True, alpha=0.3)
        
        # Pendulum angle
        plt.subplot(3, 2, 3)
        plt.plot(self.Y[:, 0], self.Y[:, 1], '-ob', markersize=3)
        plt.xlabel('Angle (rad)')
        plt.ylabel('Angular Velocity (rad/s)')
        plt.title('Phase-phase plot')
        plt.grid(True, alpha=0.3)
        
        plt.suptitle('Pendulum Swing-Up (MuJoCo)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_to_path is not None:
            plt.savefig(save_to_path, format='png', dpi=150)
            print(f"Figure saved to: {save_to_path}")
        
        plt.show()


if __name__ == '__main__':
    # Setup paths
    cwd = Path.cwd()
    file_loc = cwd / "examples/inverted_pendulum_mujoco"
    file_loc.mkdir(parents=True, exist_ok=True)
    
    model_path = file_loc / "models/pendulum.xml"
    save_path = file_loc / "models/pendulum_trajectory.png"
    
    # CartPole parameters
    nx, nu, dt = 2, 1, 0.1
    
    # Initial state: [x, x_dot, theta, theta_dot]
    # Pole hanging down
    y_init = np.array([np.pi, 0.0])
    
    # Time horizon
    time_horizon = 30
    
    # Test control (zero or oscillating)
    # control = np.zeros(time_horizon)
    control = 15.0 * np.sin(np.linspace(0, 4*np.pi, time_horizon))
    
    print('='*70)
    print('CARTPOLE MUJOCO SIMULATION')
    print('='*70)
    print(f'Model file: {model_path}')
    print(f'Initial state: {y_init}')
    print(f'θ={np.degrees(y_init[0]):.1f}°, θ̇={y_init[1]:.2f}rad/s]')
    print(f'Horizon: {time_horizon} steps ({time_horizon*dt:.1f}s)')
    print('='*70 + '\n')
    
    # Create MuJoCo simulator
    sim = SimulatePendulum(nx, nu, dt, model_path=str(model_path))
    
    # Run simulation
    trajectory = sim.simulate_trajectory(
        y_init=y_init,
        u=control,
        horizon=time_horizon
    )
    
    # Print final state
    final_state = trajectory[-1]
    print('\nFinal state:')
    print(f'[θ={np.degrees(final_state[0]):.2f}°, θ̇={final_state[1]:.4f}rad/s]')
    
    theta_error = np.abs(final_state[0])
    success = theta_error < np.deg2rad(5)
    print(f'\nAngle error: {np.degrees(theta_error):.2f}°')
    print(f'Success: {"YES ✓" if success else "NO ✗"}')
    print('='*70 + '\n')
    
    # Draw figure
    sim.draw_figure(save_to_path=save_path)