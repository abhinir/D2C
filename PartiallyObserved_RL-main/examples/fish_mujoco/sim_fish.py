import numpy as np
import mujoco
import matplotlib.pyplot as plt
from pathlib import Path

class SimulateFish:
    def __init__(self, nx, nu, dt, model_path="fish.xml", n_substeps=5):
        """
        Initialize MuJoCo Fish simulator
        
        Args:
            nx: state dimension (27)
            nu: control dimension (5)
            dt: timestep (0.1s for control, subdivided into n_substeps)
            model_path: path to MuJoCo XML file
            n_substeps: number of simulation substeps per control step
        """
        self.nx = nx
        self.nu = nu
        self.dt = dt
        self.n_substeps = n_substeps
        self.dt_substep = dt / n_substeps
        
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Set substep timestep
        self.model.opt.timestep = self.dt_substep
        
        
        # Verify dimensions
        print(f"Model nq (position dim): {self.model.nq}")
        print(f"Model nv (velocity dim): {self.model.nv}")
        print(f"Total state dim (nq + nv): {self.model.nq + self.model.nv}")
        print(f"Number of actuators: {self.model.nu}")
        
    def reset(self, init_state=None):
        """
        Reset simulation to initial state
        
        Args:
            init_state: Full state vector [qpos (14), qvel (13)] = 27 dimensions
                       qpos: [x, y, z, qw, qx, qy, qz, tail1, tail_twist, tail2, 
                              finright_roll, finright_pitch, finleft_roll, finleft_pitch]
                       qvel: [vx, vy, vz, wx, wy, wz, tail1_dot, tail_twist_dot, tail2_dot,
                              finright_roll_dot, finright_pitch_dot, finleft_roll_dot, finleft_pitch_dot]
        """
        mujoco.mj_resetData(self.model, self.data)
        
        if init_state is not None:
            # Set positions (14 DOF: 7 for free joint + 7 actuated joints)
            nq = self.model.nq
            self.data.qpos[:nq] = init_state[:nq]
            
            # Set velocities (13 DOF: 6 for free joint + 7 joint velocities)
            nv = self.model.nv
            self.data.qvel[:nv] = init_state[nq:nq+nv]
        else:
            # Default: fish at origin, upright orientation
            self.data.qpos[0:3] = [0, 0, 0.1]  # x, y, z position
            self.data.qpos[3:7] = [1, 0, 0, 0]  # quaternion (w, x, y, z)
            # All joint angles initialized to zero by reset
        
        # Forward to update derived quantities
        mujoco.mj_forward(self.model, self.data)
    
    def get_state(self):
        """
        Get current state as [qpos (14), qvel (13)] = 27 dimensions
        """
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        
        return np.concatenate([qpos, qvel])
    
    def get_position(self):
        """Get fish center of mass position [x, y, z]"""
        return self.data.qpos[0:3].copy()
    
    def get_orientation(self):
        """Get fish orientation as quaternion [qw, qx, qy, qz]"""
        return self.data.qpos[3:7].copy()
    
    def get_velocity(self):
        """Get fish linear velocity [vx, vy, vz]"""
        return self.data.qvel[0:3].copy()
    
    def get_angular_velocity(self):
        """Get fish angular velocity [wx, wy, wz]"""
        return self.data.qvel[3:6].copy()
    
    def step(self, u):
        """
        Take one control step with n_substeps simulation substeps
        
        Args:
            u: control array [tail, tail_twist, fins_flap, finleft_pitch, finright_pitch]
        """
        # Set control (held constant for all substeps)
        for i in range(min(len(u), self.nu)):
            self.data.ctrl[i] = u[i]
        
        # Step simulation n_substeps times
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        
        return self.get_state()
    
    def simulate_trajectory(self, y_init=None, u=None, horizon=600):
        """
        Simulate Fish trajectory
        
        Args:
            y_init: Initial state (27-dim, if None uses default)
            u: Control sequence [horizon x 5]
            horizon: Number of control timesteps
        
        Returns:
            Y: State trajectory [horizon+1 x 27]
        """
        if u is None:
            u = np.zeros((horizon, self.nu))
        elif u.shape[0] != horizon:
            u = np.zeros((horizon, self.nu))
        
        # Reset to initial state
        self.reset(y_init)
        
        # Storage
        self.T = np.arange(horizon + 1) * self.dt
        self.Y = np.zeros((horizon + 1, self.nx))
        self.U = u
        
        # Store positions for trajectory visualization
        self.positions = np.zeros((horizon + 1, 3))
        self.velocities = np.zeros((horizon + 1, 3))
        self.orientations = np.zeros((horizon + 1, 4))
        
        # Store initial state
        self.Y[0, :] = self.get_state()
        self.positions[0, :] = self.get_position()
        self.velocities[0, :] = self.get_velocity()
        self.orientations[0, :] = self.get_orientation()
        
        # Simulate forward
        for i in range(horizon):
            state = self.step(u[i])
            self.Y[i + 1, :] = state
            self.positions[i + 1, :] = self.get_position()
            self.velocities[i + 1, :] = self.get_velocity()
            self.orientations[i + 1, :] = self.get_orientation()
        
        return self.Y
    
    def draw_figure(self, save_to_path=None):
        """Plot trajectory results"""
        fig = plt.figure(figsize=(16, 12))
        
        # X position
        plt.subplot(3, 3, 1)
        plt.plot(self.T, self.positions[:, 0], '-ob', markersize=2)
        plt.xlabel('Time (s)')
        plt.ylabel('X Position (m)')
        plt.title('X Position')
        plt.grid(True, alpha=0.3)
        
        # Y position
        plt.subplot(3, 3, 2)
        plt.plot(self.T, self.positions[:, 1], '-og', markersize=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Y Position (m)')
        plt.title('Y Position')
        plt.grid(True, alpha=0.3)
        
        # Z position
        plt.subplot(3, 3, 3)
        plt.plot(self.T, self.positions[:, 2], '-or', markersize=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Z Position (m)')
        plt.title('Z Position')
        plt.grid(True, alpha=0.3)
        
        # XY trajectory (top view)
        plt.subplot(3, 3, 4)
        plt.plot(self.positions[:, 0], self.positions[:, 1], '-ob', markersize=2)
        plt.plot(self.positions[0, 0], self.positions[0, 1], 'go', markersize=10, label='Start')
        plt.plot(self.positions[-1, 0], self.positions[-1, 1], 'ro', markersize=10, label='End')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title('XY Trajectory (Top View)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # XZ trajectory (side view)
        plt.subplot(3, 3, 5)
        plt.plot(self.positions[:, 0], self.positions[:, 2], '-ob', markersize=2)
        plt.plot(self.positions[0, 0], self.positions[0, 2], 'go', markersize=10, label='Start')
        plt.plot(self.positions[-1, 0], self.positions[-1, 2], 'ro', markersize=10, label='End')
        plt.xlabel('X Position (m)')
        plt.ylabel('Z Position (m)')
        plt.title('XZ Trajectory (Side View)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3D trajectory
        ax = fig.add_subplot(3, 3, 6, projection='3d')
        ax.plot(self.positions[:, 0], self.positions[:, 1], self.positions[:, 2], '-b', linewidth=2)
        ax.scatter(self.positions[0, 0], self.positions[0, 1], self.positions[0, 2], c='g', s=100, label='Start')
        ax.scatter(self.positions[-1, 0], self.positions[-1, 1], self.positions[-1, 2], c='r', s=100, label='End')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D Trajectory')
        ax.legend()
        
        # Linear velocities
        plt.subplot(3, 3, 7)
        plt.plot(self.T, self.velocities[:, 0], '-r', label='vx', linewidth=1.5)
        plt.plot(self.T, self.velocities[:, 1], '-g', label='vy', linewidth=1.5)
        plt.plot(self.T, self.velocities[:, 2], '-b', label='vz', linewidth=1.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (m/s)')
        plt.title('Linear Velocities')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Control inputs
        plt.subplot(3, 3, 8)
        for i in range(self.nu):
            plt.plot(self.T[:-1], self.U[:, i], linewidth=1.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Control Input')
        plt.title('Control Inputs')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        
        # Speed over time
        plt.subplot(3, 3, 9)
        speed = np.linalg.norm(self.velocities, axis=1)
        plt.plot(self.T, speed, '-ob', markersize=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Speed (m/s)')
        plt.title('Speed')
        plt.grid(True, alpha=0.3)
        
        plt.suptitle('Fish Robot Swimming (MuJoCo)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_to_path is not None:
            plt.savefig(save_to_path, format='png', dpi=150)
            print(f"Figure saved to: {save_to_path}")
        
        plt.show()


if __name__ == '__main__':
    # Setup paths
    cwd = Path.cwd()
    file_loc = cwd / "examples/fish_mujoco"
    file_loc.mkdir(parents=True, exist_ok=True)
    
    model_path = file_loc / "models/fish.xml"
    save_path = file_loc / "models/fish_trajectory.png"
    
    # Fish parameters from your specification
    state_dimension = 27
    control_dimension = 6
    obs_dimension = state_dimension
    
    # Simulation parameters
    dt = 0.1
    horizon = 600
    n_substeps = 5
    ctrl_state_freq_ratio = 1
    
    
    # Test control
    control = 0.1*np.random.normal(0, 1, (horizon, 6))
    
    
    
    print('='*70)
    print('FISH ROBOT MUJOCO SIMULATION')
    print('='*70)
    print(f'Model file: {model_path}')
    print(f'State dimension: {state_dimension}')
    print(f'Control dimension: {control_dimension}')
    print(f'Control timestep: {dt}s')
    print(f'Simulation substeps: {n_substeps}')
    print(f'Substep timestep: {dt/n_substeps}s')
    print(f'Horizon: {horizon} steps ({horizon*dt:.1f}s)')
    print('='*70 + '\n')
    
    # Create MuJoCo simulator
    sim = SimulateFish(state_dimension, control_dimension, dt, 
                       model_path=str(model_path), n_substeps=n_substeps)
    
    # Run simulation
    trajectory = sim.simulate_trajectory(
        y_init=None,  # Use default starting position
        u=control,
        horizon=horizon
    )
    
    # Print final state
    final_pos = sim.positions[-1]
    final_vel = sim.velocities[-1]
    distance_traveled = np.linalg.norm(sim.positions[-1, :2] - sim.positions[0, :2])
    
    print('\nFinal state:')
    print(f'Position: [{final_pos[0]:.4f}, {final_pos[1]:.4f}, {final_pos[2]:.4f}] m')
    print(f'Velocity: [{final_vel[0]:.4f}, {final_vel[1]:.4f}, {final_vel[2]:.4f}] m/s')
    print(f'Distance traveled (XY): {distance_traveled:.4f} m')
    print(f'Average speed: {distance_traveled/(horizon*dt):.4f} m/s')
    print('='*70 + '\n')
    
    # Draw figure
    sim.draw_figure(save_to_path=save_path)