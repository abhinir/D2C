import numpy as np
import mujoco
import matplotlib.pyplot as plt
from pathlib import Path

class SimulateGo2:
    def __init__(self, nx, nu, dt, model_path="go2.xml"):
        """
        Initialize MuJoCo Unitree Go2 simulator
        
        Args:
            nx: state dimension (37 for full state: 7 base pos/quat + 12 joints + 6 base vel + 12 joint vel)
            nu: control dimension (12 joint torques)
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
        
        # Go2 has 12 actuated joints (3 per leg: hip, thigh, calf)
        # Joint order: FL, FR, RL, RR (Front-Left, Front-Right, Rear-Left, Rear-Right)
        self.joint_names = [
            'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
            'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
            'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
            'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint'
        ]
        
        # Default standing configuration from keyframe [1]
        self.standing_qpos = np.array([0, 0.9, -1.8] * 4)  # Hip, thigh, calf for each leg
        
    def reset(self, init_state=None):
        """
        Reset simulation to initial state
        
        Args:
            init_state: Full state vector [37] or None for default standing pose
                      [base_pos(3), base_quat(4), joint_pos(12), 
                       base_vel(3), base_angvel(3), joint_vel(12)]
        """
        mujoco.mj_resetData(self.model, self.data)
        
        if init_state is None:
            # Use default standing pose from keyframe [1]
            # Base position at height 0.27m
            self.data.qpos[0:3] = [0, 0, 0.27]  # x, y, z
            self.data.qpos[3:7] = [1, 0, 0, 0]  # quaternion (w, x, y, z)
            self.data.qpos[7:19] = self.standing_qpos  # joint positions
            self.data.qvel[:] = 0  # all velocities zero
        else:
            # Set from provided state
            self.data.qpos[0:3] = init_state[0:3]      # base position
            self.data.qpos[3:7] = init_state[3:7]      # base quaternion
            self.data.qpos[7:19] = init_state[7:19]    # joint positions
            self.data.qvel[0:3] = init_state[19:22]    # base linear velocity
            self.data.qvel[3:6] = init_state[22:25]    # base angular velocity
            self.data.qvel[6:18] = init_state[25:37]   # joint velocities
        
        # Forward to update derived quantities
        mujoco.mj_forward(self.model, self.data)
    
    def get_state(self):
        """
        Get current state 
        
        Returns:
            state: [37] array containing:
                   [base_pos(3), base_quat(4), joint_pos(12),
                    base_vel(3), base_angvel(3), joint_vel(12)]
        """
        state = np.zeros(self.nx)
        
        # Positions
        state[0:3] = self.data.qpos[0:3]      # base position
        state[3:7] = self.data.qpos[3:7]      # base quaternion
        state[7:19] = self.data.qpos[7:19]    # joint positions
        
        # Velocities
        state[19:22] = self.data.qvel[0:3]    # base linear velocity
        state[22:25] = self.data.qvel[3:6]    # base angular velocity
        state[25:37] = self.data.qvel[6:18]   # joint velocities
        
        return state
    
    def step(self, u):
        """
        Take one simulation step with control input
        
        Args:
            u: control torques [12] for all joints
        """
        # Set control (clipped to actuator limits from XML [1])
        # Most joints: ±23.7 Nm, knee joints: ±45.43 Nm
        self.data.ctrl[:] = np.clip(u, -45.43, 45.43)
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        
        return self.get_state()
    
    def simulate_quadruped(self, x_init, u_scalar):
        """
        Single step simulation (for iLQR compatibility)
        
        Args:
            x_init: Initial state [37] (can be flattened array)
            u_scalar: Control input [12] (can be flattened array)
        
        Returns:
            next_state: Next state [37]
        """
        x_init = np.array(x_init).flatten()
        u_scalar = np.array(u_scalar).flatten()
        
        self.reset(x_init)
        next_state = self.step(u_scalar)
        
        return next_state
    
    def simulate_trajectory(self, y_init=None, u=None, horizon=1):
        """
        Simulate Go2 trajectory
        
        Args:
            y_init: Initial state [37] or None for default standing
            u: Control sequence [horizon x 12]
            horizon: Number of timesteps
        
        Returns:
            Y: State trajectory [horizon+1 x 37]
        """
        if u is None:
            u = np.zeros((horizon, self.nu))
        
        if u.shape[0] != horizon:
            u = np.zeros((horizon, self.nu))
        
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
            state = self.step(u[i])
            self.Y[i + 1, :] = state
        
        return self.Y
    
   
    def simulate_trajectory_PID(self, y_init=None, u=None, horizon=1,KP=50.0, KD=1.0):
        """
        Simulate Go2 trajectory
        
        Args:
            y_init: Initial state [37] or None for default standing
            u: Control sequence [horizon x 12]
            horizon: Number of timesteps
        
        Returns:
            Y: State trajectory [horizon+1 x 37]
        """
        if u is None:
            u = np.zeros((horizon, self.nu))
        
        if u.shape[0] != horizon:
            u = np.zeros((horizon, self.nu))
        
        # Reset to initial state
        self.reset(y_init)
        
        # Storage
        self.T = np.arange(horizon + 1) * self.dt
        self.Y = np.zeros((horizon + 1, self.nx))
        self.U = u
        
        # Store initial state
        self.Y[0, :] = self.get_state()
        
        u = np.concatenate((u, np.zeros((1,self.nu))), axis=0)  # Extra step for last control
        # Simulate forward
        for i in range(horizon):
            state = self.step(u[i])
            self.Y[i + 1, :] = state
            u[i+1] = KP * (q_des - state[7:19]) + KD * (0.0 - state[25:37])

        self.U = u[0:horizon,:]
        return self.Y
    
    def draw_figure(self, save_to_path=None):
        """Plot trajectory results"""
        fig = plt.figure(figsize=(16, 12))
        
        # Base position
        plt.subplot(4, 3, 1)
        plt.plot(self.T, self.Y[:, 0], '-r', label='x', linewidth=2)
        plt.plot(self.T, self.Y[:, 1], '-g', label='y', linewidth=2)
        plt.plot(self.T, self.Y[:, 2], '-b', label='z', linewidth=2)
        plt.axhline(y=0.27, color='b', linestyle='--', alpha=0.5, label='target height')
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        plt.title('Base Position')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Base orientation (convert quaternion to euler for visualization)
        plt.subplot(4, 3, 2)
        euler_angles = np.array([self._quat_to_euler(self.Y[i, 3:7]) for i in range(len(self.Y))])
        plt.plot(self.T, np.degrees(euler_angles[:, 0]), '-r', label='roll', linewidth=2)
        plt.plot(self.T, np.degrees(euler_angles[:, 1]), '-g', label='pitch', linewidth=2)
        plt.plot(self.T, np.degrees(euler_angles[:, 2]), '-b', label='yaw', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (deg)')
        plt.title('Base Orientation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Base velocities
        plt.subplot(4, 3, 3)
        plt.plot(self.T, self.Y[:, 19], '-r', label='vx', linewidth=2)
        plt.plot(self.T, self.Y[:, 20], '-g', label='vy', linewidth=2)
        plt.plot(self.T, self.Y[:, 21], '-b', label='vz', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (m/s)')
        plt.title('Base Linear Velocity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Joint positions for each leg
        leg_names = ['Front-Left', 'Front-Right', 'Rear-Left', 'Rear-Right']
        joint_labels = ['Hip', 'Thigh', 'Calf']
        
        for leg_idx in range(4):
            plt.subplot(4, 3, 4 + leg_idx)
            for j in range(3):
                joint_idx = 7 + leg_idx * 3 + j
                plt.plot(self.T, self.Y[:, joint_idx], linewidth=2, label=joint_labels[j])
            plt.xlabel('Time (s)')
            plt.ylabel('Angle (rad)')
            plt.title(f'{leg_names[leg_idx]} Joint Positions')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Control inputs
        plt.subplot(4, 3, 10)
        for i in range(0, 12):  # Plot every 3rd control for clarity
            plt.plot(self.T[:-1], self.U[:, i], linewidth=2, alpha=0.7)
        plt.xlabel('Time (s)')
        plt.ylabel('Torque (Nm)')
        plt.title('Control Inputs (Hip joints)')
        plt.grid(True, alpha=0.3)
        
        # 3D trajectory
        ax = fig.add_subplot(4, 3, 11, projection='3d')
        ax.plot(self.Y[:, 0], self.Y[:, 1], self.Y[:, 2], '-b', linewidth=2)
        ax.scatter(self.Y[0, 0], self.Y[0, 1], self.Y[0, 2], c='g', s=100, label='Start')
        ax.scatter(self.Y[-1, 0], self.Y[-1, 1], self.Y[-1, 2], c='r', s=100, label='End')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D Base Trajectory')
        ax.legend()
        
        plt.suptitle('Unitree Go2 Simulation (MuJoCo)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_to_path is not None:
            plt.savefig(save_to_path, format='png', dpi=150)
            print(f"Figure saved to: {save_to_path}")
        
        plt.show()
    
    def _quat_to_euler(self, quat):
        """Convert quaternion (w,x,y,z) to euler angles (roll, pitch, yaw)"""
        w, x, y, z = quat
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sinp, -1, 1))
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw])


if __name__ == '__main__':
    # Setup paths
    cwd = Path.cwd()
    file_loc = cwd / "examples/quadruped/models"
    file_loc.mkdir(parents=True, exist_ok=True)
    
    model_path = file_loc / "scene.xml"
    save_path = file_loc / "go2_trajectory.png"
    
    # Go2 parameters
    nx, nu, dt = 37, 12, 0.01  # 37 states, 12 controls, 10ms timestep
    
    # Time horizon
    time_horizon = 1000  # 10 seconds
   
    # position actuators (XML currently uses <position> actuators)
    # control = np.tile(np.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8]), (time_horizon,1))
    
    print('='*70)
    print('UNITREE GO2 MUJOCO SIMULATION')
    print('='*70)
    print(f'Model file: {model_path}')
    print(f'State dimension: {nx} (base pose + 12 joints + velocities)')
    print(f'Control dimension: {nu} (12 joint torques)')
    print(f'Horizon: {time_horizon} steps ({time_horizon*dt:.2f}s)')
    print('='*70 + '\n')
    
    # Create MuJoCo simulator
    sim = SimulateGo2(nx, nu, dt, model_path=str(model_path))
    
    # Desired joint positions (rad) for 12 joints (FL, FR, RL, RR: Hip, Thigh, Calf)
    q_des = np.array([0., 0.9, -1.8, 0., 0.9, -1.8, 0., 0.9, -1.8, 0., 0.9, -1.8])

    # PD gains (for mapping desired positions -> torques)
    KP = 200
    KD = 1.0

    control = np.zeros((time_horizon, nu))

    # Run simulation from default standing pose with PID control
    trajectory = sim.simulate_trajectory_PID(
        y_init=None,  # Use default standing pose
        u=control,
        horizon=time_horizon,
        KP=KP, KD=KD)
    
    # Print final state summary
    final_state = trajectory[-1]
    print('\nFinal state summary:')
    print(f'Base position: [{final_state[0]:.3f}, {final_state[1]:.3f}, {final_state[2]:.3f}] m')
    print(f'Base height: {final_state[2]:.3f} m')
    print(f'Base velocity: {np.linalg.norm(final_state[19:22]):.3f} m/s')
    print(f'Final joint torques: {sim.U[-1]} Nm')
    print('='*70 + '\n')
    
    # Draw figure
    sim.draw_figure(save_to_path=save_path)