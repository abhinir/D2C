import numpy as np
import sys
import os
from pathlib import Path
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from sim_cartpole2 import SimulateCartPole
from cartpole_params import *
from main_ilqr import iLQR
from ltv_sys_id import LTV_SysID


class RunCartPoleNoisy(SimulateCartPole):
    """CartPole with additive process noise matching MATLAB implementation"""
    def __init__(self, state_dimension, control_dimension, dt, noise_epsilon=0.0):
        SimulateCartPole.__init__(self, state_dimension, control_dimension, dt)
        self.noise_epsilon = noise_epsilon
        # State covariance matching MATLAB: diag([0.4321, 2.8743, 3.6426, 1.5945])
        self.state_cov = np.array([0.4321, 2.8743, 3.6426, 1.5945])
    
    def simulate(self, x, u):
        """
        Simulate one step with process noise
        Process noise: w = epsilon * sqrt(dt) * N(0, Cov)
        Applied as: state_new = X_out + w
        """
        # Integrate with clean control (no control noise)
        x_flat = np.array(x).flatten()
        u_flat = np.array(u).flatten()
        next_state = self.simulate_cartpole(x_flat, u_flat)[-1]
        
        # Ensure next_state is (4,) shape
        next_state = next_state.flatten()
        
        # Add process noise AFTER integration
        if self.noise_epsilon > 0:
            w_scale = self.noise_epsilon * np.sqrt(self.dt)
            std_devs = np.sqrt(self.state_cov)
            
            # Create noise with shape (4,)
            noise = np.random.normal(0, 1, size=4) * std_devs
            next_state = next_state + w_scale * noise
        
        return next_state


class RunCartPole(SimulateCartPole):
    """CartPole without noise (for planning)"""
    def __init__(self, state_dimension, control_dimension, dt):
        SimulateCartPole.__init__(self, state_dimension, control_dimension, dt)
    
    def simulate(self, x, u):
        x_flat = np.array(x).flatten()
        u_flat = np.array(u).flatten()
        return self.simulate_cartpole(x_flat, u_flat)[-1]


def shrinking_horizon_ilqr(model, init_state, final_state, total_horizon, 
                           planning_horizon, n_ilqr_iterations_per_step=10):
    """
    Shrinking horizon iLQR implementation with noise execution
    
    Args:
        model: Dynamics model (with noise for execution)
        init_state: Initial state [nx, 1]
        final_state: Goal state [nx, 1]
        total_horizon: Total execution horizon
        planning_horizon: Planning horizon
        n_ilqr_iterations_per_step: iLQR iterations per step
    
    Returns:
        state_trajectory: Executed state trajectory
        control_trajectory: Executed control trajectory
    """
    state_trajectory = [init_state.copy()]
    control_trajectory = []
    current_state = init_state.copy()
    
    for step in range(total_horizon):
        remaining_horizon = total_horizon - step
        current_planning_horizon = min(planning_horizon, remaining_horizon)
        
        # Warm start with previous solution
        if step > 0 and 'ilqr' in locals() and hasattr(ilqr, 'U'):
            u_init = np.zeros((current_planning_horizon, control_dimension, 1))
            prev_u_length = min(len(ilqr.U) - 1, current_planning_horizon)
            if prev_u_length > 0:
                u_init[:prev_u_length] = ilqr.U[1:prev_u_length+1]
        else:
            u_init = np.load('cartpole_optimal.npy')
        
        # Create clean model for planning (no noise)
        clean_model = RunCartPole(state_dimension, control_dimension, dt)
        
        # Create iLQR instance
        ilqr = iLQR(
            clean_model, 
            state_dimension, 
            control_dimension, 
            alpha, 
            current_planning_horizon,
            current_state,
            final_state, 
            Q, 
            Q_final, 
            R, 
            nominal_init_stddev,
            n_sys_id_samples=40, 
            pert_sys_id_sigma=1e-5, 
            arma_sys_id_flag=False
        )
        
        # Run iLQR optimization
        ilqr.iterate_ilqr(n_ilqr_iterations_per_step, u_init=u_init)
        
        # Extract first control action
        optimal_control = ilqr.U[0].copy()
        
        # Execute with noisy model
        next_state = model.simulate(current_state.flatten(), optimal_control.flatten())
        next_state = next_state.reshape(-1, 1)
        
        # Store trajectory
        control_trajectory.append(optimal_control)
        state_trajectory.append(next_state)
        current_state = next_state
    
    return np.array(state_trajectory), np.array(control_trajectory)


def compute_trajectory_cost(state_traj, control_traj, init_state, final_state, Q, Q_final, R):
    """Compute total cost of trajectory"""
    horizon = len(control_traj)
    running_cost = 0.0
    
    # Initial cost
    x_err = init_state - final_state
    u = control_traj[0]
    running_cost += float((x_err.T @ Q @ x_err)[0, 0])
    running_cost += float((u.T @ R @ u)[0, 0])
    
    # Running costs
    for t in range(horizon - 1):
        x_err = state_traj[t + 1] - final_state
        u = control_traj[t + 1]
        running_cost += float((x_err.T @ Q @ x_err)[0, 0])
        running_cost += float((u.T @ R @ u)[0, 0])
    
    # Terminal cost
    x_err_final = state_traj[-1] - final_state
    terminal_cost = float((x_err_final.T @ Q_final @ x_err_final)[0, 0])
    
    return running_cost + terminal_cost


def epsilon_sweep(epsilon_values, num_runs=100, total_horizon=30, 
                 planning_horizon=30, n_ilqr_iterations_per_step=50):
    """
    Sweep over epsilon values and save mean/std of total cost
    
    Args:
        epsilon_values: List of epsilon values
        num_runs: Number of Monte Carlo runs per epsilon
        total_horizon: Total execution horizon
        planning_horizon: Planning horizon
        n_ilqr_iterations_per_step: iLQR iterations per step
    
    Returns:
        results: Dictionary with statistics per epsilon
    """
    print('='*70)
    print('PROCESS NOISE SWEEP: CARTPOLE SHRINKING HORIZON iLQR')
    print('='*70)
    print(f'Epsilon values: {epsilon_values}')
    print(f'Runs per epsilon: {num_runs}')
    print(f'Noise model: state_new = X_out + ε*sqrt(dt)*N(0,Cov)')
    print('='*70 + '\n')
    
    # Initial and goal states
    init_state = np.zeros((state_dimension, 1))
    init_state[2] = np.pi  # Pole hanging down
    final_state = np.zeros((state_dimension, 1))
    
    results = {}
    
    for epsilon in epsilon_values:
        print(f'\n{"="*70}')
        print(f'Processing epsilon = {epsilon}')
        print(f'{"="*70}')
        
        costs = []
        success_count = 0
        failed_count = 0
        
        for run in range(num_runs):
            # Create noisy model
            noisy_model = RunCartPoleNoisy(
                state_dimension, 
                control_dimension, 
                dt, 
                noise_epsilon=epsilon
            )
            
            try:
                # Run shrinking horizon iLQR
                state_traj, control_traj = shrinking_horizon_ilqr(
                    model=noisy_model,
                    init_state=init_state.copy(),
                    final_state=final_state,
                    total_horizon=total_horizon,
                    planning_horizon=planning_horizon,
                    n_ilqr_iterations_per_step=n_ilqr_iterations_per_step
                )

                # np.save('cartpole_optimal.npy', control_traj)
                
                # Compute cost
                cost = compute_trajectory_cost(
                    state_traj, control_traj, init_state, final_state, Q, Q_final, R
                )
                
                # Filter invalid costs (NaN, Inf, or > 50000)
                if not np.isnan(cost) and not np.isinf(cost) and 0 < cost < 50000:
                    costs.append(cost)
                    
                    # Check success
                    final_error = np.linalg.norm(state_traj[-1] - final_state)
                    if final_error < 0.3:
                        success_count += 1
                else:
                    failed_count += 1
                    
            except Exception as e:
                failed_count += 1
                if run % 20 == 0:
                    print(f'  Run {run} failed: {str(e)[:50]}')
            
            if (run + 1) % 20 == 0:
                print(f'  Completed {run + 1}/{num_runs} runs (Valid: {len(costs)}, Failed: {failed_count})')
        
        # Compute statistics on valid costs
        if len(costs) > 0:
            mean_cost = float(np.mean(costs))
            std_cost = float(np.std(costs))
            success_rate = success_count / num_runs
        else:
            mean_cost = np.nan
            std_cost = np.nan
            success_rate = 0.0
        
        results[epsilon] = {
            'mean': mean_cost,
            'std': std_cost,
            'valid_runs': len(costs),
            'failed_runs': failed_count,
            'total_runs': num_runs,
            'success_rate': success_rate
        }
        
        print(f'\n  Results for epsilon = {epsilon}:')
        print(f'    Valid runs:   {len(costs)}/{num_runs}')
        print(f'    Mean cost:    {mean_cost:.4f}')
        print(f'    Std cost:     {std_cost:.4f}')
        print(f'    Success rate: {100*success_rate:.1f}%')
    
    return results


if __name__ == "__main__":
    cwd = os.getcwd()
    path_to_export = Path(cwd)/"examples/cartpole/Cartpole_Experiments/epsilon_sweep"
    path_to_export.mkdir(parents=True, exist_ok=True)
    
    # Define epsilon values
    epsilon_values = [0.0]#[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    
    # Run epsilon sweep
    results = epsilon_sweep(
        epsilon_values=epsilon_values,
        num_runs=1,
        total_horizon=30,
        planning_horizon=30,
        n_ilqr_iterations_per_step=50
    )
    
    # Save results to JSON
    save_path = path_to_export / "epsilon_sweep_results.json"
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary table
    print('\n' + '='*70)
    print('EPSILON SWEEP RESULTS')
    print('='*70)
    print(f'{"Epsilon":<10} {"Mean Cost":<15} {"Std Cost":<15} {"Valid":<10} {"Success":<10}')
    print('-'*70)
    for eps, stats in results.items():
        print(f'{float(eps):<10.1f} {stats["mean"]:<15.4f} {stats["std"]:<15.4f} '
              f'{stats["valid_runs"]:<10} {100*stats["success_rate"]:<9.1f}%')
    print('='*70)
    
    print(f'\nResults saved to: {save_path}')