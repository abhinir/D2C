import numpy as np
import math
import sys
import os
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from sim_cartpole2 import SimulateCartPole
from cartpole_params import *
from main_ilqr import iLQR
from ltv_sys_id import LTV_SysID

class RunCartPoleNoisy(SimulateCartPole):
    """CartPole with additive control noise"""
    def __init__(self, state_dimension, control_dimension, dt, noise_epsilon=0.0):
        SimulateCartPole.__init__(self, state_dimension, control_dimension, dt)
        self.noise_epsilon = noise_epsilon
    
    def simulate(self, x, u):
        # Add control noise: u += epsilon * 20 * N(0,1)
        if self.noise_epsilon > 0:
            noise = np.random.normal(0, 1, size=u.shape)
            u_noisy = u + self.noise_epsilon * 5.0 * noise
        else:
            u_noisy = u
        return self.simulate_cartpole(x, u_noisy)[-1]

class RunCartPole(SimulateCartPole):
    def __init__(self, state_dimension, control_dimension, dt):
        SimulateCartPole.__init__(self, state_dimension, control_dimension, dt)
    
    def simulate(self, x, u):
        return self.simulate_cartpole(x, u)[-1]

def shrinking_horizon_ilqr(model, init_state, final_state, total_horizon, 
                           planning_horizon, n_ilqr_iterations_per_step=10):
    """Shrinking horizon iLQR implementation"""
    state_trajectory = [init_state.copy()]
    control_trajectory = []
    current_state = init_state.copy()
    
    for step in range(total_horizon):
        remaining_horizon = total_horizon - step
        current_planning_horizon = min(planning_horizon, remaining_horizon)
        
        # Warm start
        if step > 0 and hasattr(ilqr, 'U') and len(control_trajectory) > 0:
            u_init = np.zeros((current_planning_horizon, control_dimension, 1))
            prev_u_length = min(len(ilqr.U) - 1, current_planning_horizon)
            if prev_u_length > 0:
                u_init[:prev_u_length] = ilqr.U[1:prev_u_length+1]
        else:
            u_init = None#np.load('examples/cartpole/u_init_smart.npy').reshape((30, 1, 1))
        
        # Create clean model for planning
        clean_model = RunCartPole(state_dimension, control_dimension, dt)
        
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
        # if step>0:
        #     n_ilqr_iterations_per_step = 10
        ilqr.iterate_ilqr(n_ilqr_iterations_per_step, u_init=u_init)
        
        # Extract first control
        optimal_control = ilqr.U[0].copy()
        
        # Execute with noisy model
        next_state = model.simulate(current_state.flatten(), optimal_control.flatten())
        next_state = next_state.reshape(-1, 1)
        
        # Store trajectory
        control_trajectory.append(optimal_control)
        state_trajectory.append(next_state)
        current_state = next_state
        
        # Early termination
        state_error = np.linalg.norm(current_state - final_state)
        # if state_error < 0.1:
        #     break
    
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


def run_monte_carlo(epsilon, num_runs=100, total_horizon=30, 
                   planning_horizon=30, n_ilqr_iterations_per_step=50):
    """
    Run multiple simulations with fixed epsilon and compute statistics
    
    Args:
        epsilon: Control noise level
        num_runs: Number of Monte Carlo runs
        total_horizon: Total execution horizon
        planning_horizon: Planning horizon for iLQR
        n_ilqr_iterations_per_step: iLQR iterations per step
    
    Returns:
        mean_cost: Mean total cost
        std_cost: Standard deviation of total cost
        costs: Array of all costs
    """
    print(f'\n{"="*70}')
    print(f'Running {num_runs} simulations with epsilon = {epsilon}')
    print(f'{"="*70}')
    
    # Initial and goal states
    init_state = np.zeros((state_dimension, 1))
    init_state[2] = np.pi  # Pole hanging down
    final_state = np.zeros((state_dimension, 1))
    
    costs = []
    success_count = 0
    
    for run in range(num_runs):
        # Set random seed for reproducibility
        # np.random.seed(42 + run)
        
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
            
            # Compute cost
            cost = compute_trajectory_cost(
                state_traj, control_traj, init_state, final_state, Q, Q_final, R
            )
            costs.append(cost)
            
            # Check success
            final_error = np.linalg.norm(state_traj[-1] - final_state)
            if final_error < 0.5:
                success_count += 1
            
            if (run + 1) % 10 == 0:
                print(f'  Completed {run + 1}/{num_runs} runs...')
                
        except Exception as e:
            print(f'  Run {run} failed: {e}')
            # costs.append(np.nan)
    
    # Filter out failed runs
    valid_costs = np.array([c for c in costs if not np.isnan(c)])
    
    if len(valid_costs) > 0:
        mean_cost = np.mean(valid_costs)
        std_cost = np.std(valid_costs)
        success_rate = success_count / num_runs
    else:
        mean_cost = np.nan
        std_cost = np.nan
        success_rate = 0.0
    
    print(f'\n{"="*70}')
    print(f'RESULTS for epsilon = {epsilon}')
    print(f'{"="*70}')
    print(f'Valid runs:      {len(valid_costs)}/{num_runs}')
    print(f'Mean cost:       {mean_cost:.4f}')
    print(f'Std cost:        {std_cost:.4f}')
    print(f'Success rate:    {100*success_rate:.1f}%')
    print(f'{"="*70}\n')
    
    return mean_cost, std_cost, valid_costs


import numpy as np
import json
from pathlib import Path

def epsilon_sweep(epsilon_values, num_runs=100, total_horizon=30, 
                 planning_horizon=30, n_ilqr_iterations_per_step=50):
    """
    Sweep over epsilon values and save mean/std of total cost
    
    Args:
        epsilon_values: List of epsilon values (e.g., [0.0, 0.1, 0.2, ..., 0.6])
        num_runs: Number of Monte Carlo runs per epsilon
        total_horizon: Total execution horizon
        planning_horizon: Planning horizon
        n_ilqr_iterations_per_step: iLQR iterations per step
    
    Returns:
        results_dict: Dictionary with epsilon as key, stats as values
    """
    print('='*70)
    print('EPSILON SWEEP')
    print('='*70)
    print(f'Epsilon values: {epsilon_values}')
    print(f'Runs per epsilon: {num_runs}')
    print('='*70 + '\n')
    
    # Initial and goal states
    init_state = np.zeros((state_dimension, 1))
    init_state[2] = np.pi  # Pole hanging down
    final_state = np.zeros((state_dimension, 1))
    
    results = {}
    
    for epsilon in epsilon_values:
        print(f'\nProcessing epsilon = {epsilon}...')
        
        costs = []
        
        for run in range(num_runs):
            np.random.seed(42 + run)
            
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
                
                # Compute cost
                cost = compute_trajectory_cost(
                    state_traj, control_traj, init_state, final_state, Q, Q_final, R
                )
                costs.append(cost)
                
            except Exception as e:
                print(f'  Run {run} failed: {e}')
                costs.append(np.nan)
            
            if (run + 1) % 20 == 0:
                print(f'  Completed {run + 1}/{num_runs}')
        
        # Filter valid costs
        valid_costs = np.array([c for c in costs if not np.isnan(c)])
        
        if len(valid_costs) > 0:
            mean_cost = float(np.mean(valid_costs))
            std_cost = float(np.std(valid_costs))
        else:
            mean_cost = np.nan
            std_cost = np.nan
        
        results[epsilon] = {
            'mean': mean_cost,
            'std': std_cost,
            'valid_runs': len(valid_costs),
            'total_runs': num_runs
        }
        
        print(f'  Epsilon {epsilon}: Mean = {mean_cost:.4f}, Std = {std_cost:.4f}')
    
    return results


if __name__ == "__main__":
    cwd = os.getcwd()
    path_to_export = Path(cwd)/"examples/cartpole/Cartpole_Experiments/epsilon_sweep"
    path_to_export.mkdir(parents=True, exist_ok=True)
    
    # Define epsilon values
    epsilon_values = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    
    # Run epsilon sweep
    results = epsilon_sweep(
        epsilon_values=epsilon_values,
        num_runs=50,
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
    print(f'{"Epsilon":<10} {"Mean Cost":<15} {"Std Cost":<15} {"Valid Runs":<12}')
    print('-'*70)
    for eps, stats in results.items():
        print(f'{eps:<10.1f} {stats["mean"]:<15.4f} {stats["std"]:<15.4f} {stats["valid_runs"]:<12}')
    print('='*70)
    
    print(f'\nResults saved to: {save_path}')