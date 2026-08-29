import numpy as np
import matplotlib.pyplot as plt

class ILQR_mod:

    def __init__(self,
        dynamics: callable,
        cost: callable,
        terminal_cost: callable,
        n_state: int,
        n_control: int,
        horizon: int,
        dt: float = 0.01,
        alpha: float = 1,
        alpha_floor: float = 1e-8,
        max_iterations: int = 100,
        std_dev_per: float = 1e-2,
        num_rollouts: int = 200,
        verbose: bool = True):
        
        self.dynamics = dynamics
        self.cost = cost
        self.terminal_cost = terminal_cost
        self.n_state = n_state
        self.n_control = n_control
        self.dt = dt
        self.horizon = horizon
        self.alpha = alpha
        self.alpha_floor = alpha_floor
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        self.delta_J = 0
        self.eps = 1e-5
        self.lambda_init = 1.0
        self.lambda_max = 1e10
        self.std_dev_per = std_dev_per
        self.num_rollouts = num_rollouts
        
    def linearize_dynamics(self, x, x_next, u, index):
        
        dx = np.zeros((self.n_state, self.num_rollouts))
        du = np.zeros((self.n_control, self.num_rollouts))
        dx_next = np.zeros((self.n_state, self.num_rollouts))
        x_per = np.zeros((self.n_state, 2))
        u_per = np.zeros((self.n_control))
        for i in range(self.num_rollouts):
            dx[:,i] = np.random.normal(loc=0.0, scale = self.std_dev_per, size = self.n_state)
            du[:,i] = np.random.normal(loc=0.0, scale = self.std_dev_per, size = self.n_control)
            dx_next[:,i] = self.dynamics(x + dx[:,i], u + du[:,i], index) - x_next
            
        
        Y = dx_next
        X = np.vstack((dx, du))
        P = Y @ np.linalg.pinv(X)
        A = P[:, :self.n_state]
        B = P[:, self.n_state:]
        
        
        
        return A, B

    def quadratize_cost(self, x, u, x_prev,index):
        if u is None or index == self.horizon:
            c_x = np.zeros((self.n_state))
            c_xx = np.zeros((self.n_state, self.n_state))
            h = self.eps
            for i in range(self.n_state):
                e = np.zeros((self.n_state))
                e[i] = h
                c_x[i] = (self.terminal_cost(x + e) - self.terminal_cost(x - e))/(2*h)
            for a in range(self.n_state):
                for b in range(self.n_state):
                    e1 = np.zeros((self.n_state))
                    e1[a] = h
                    e2 = np.zeros((self.n_state))
                    e2[b] = h
                    c_xx[a, b] = self.terminal_cost(x+e1+e2) - self.terminal_cost(x+e1-e2) - self.terminal_cost(x-e1+e2) + self.terminal_cost(x-e1-e2)
                    c_xx[a, b] = c_xx[a, b]/(4*h*h)
            return c_x, c_xx
            
        c_x = np.zeros((self.n_state))
        c_u = np.zeros((self.n_control))
        h = self.eps
        for i in range(self.n_state):
            e = np.zeros((self.n_state))
            e[i] = h
            c_x[i] = (self.cost(x + e, u, x_prev) - self.cost(x - e, u, x_prev))/(2*h)
        
        for j in range(self.n_control):
            e = np.zeros((self.n_control))
            e[j] = h
            c_u[j] = (self.cost(x, u + e, x_prev) - self.cost(x, u - e, x_prev))/(2*h)
        
        c_xx = np.zeros((self.n_state, self.n_state))
        c_uu = np.zeros((self.n_control, self.n_control))
        c_ux = np.zeros((self.n_control, self.n_state))
        for a in range(self.n_state):
            for b in range(self.n_state):
                e1 = np.zeros((self.n_state))
                e1[a] = h
                e2 = np.zeros((self.n_state))
                e2[b] = h
                c_xx[a, b] = self.cost(x+e1+e2, u, x_prev) - self.cost(x+e1-e2, u, x_prev) - self.cost(x-e1+e2, u, x_prev) + self.cost(x-e1-e2, u, x_prev)
                c_xx[a, b] = c_xx[a, b]/(4*h*h)
        
        for a in range(self.n_control):
            for b in range(self.n_control):
                e1 = np.zeros((self.n_control))
                e1[a] = h
                e2 = np.zeros((self.n_control))
                e2[b] = h
                c_uu[a, b] = self.cost(x, u+e1+e2, x_prev) - self.cost(x, u+e1-e2, x_prev) - self.cost(x, u-e1+e2, x_prev) + self.cost(x, u-e1-e2, x_prev)
                c_uu[a, b] = c_uu[a, b]/(4*h*h)
        
        for a in range(self.n_control):
            for b in range(self.n_state):
                e1 = np.zeros((self.n_control))
                e1[a] = h
                e2 = np.zeros((self.n_state))
                e2[b] = h
                c_ux[a, b] = self.cost(x+e2, u+e1, x_prev) - self.cost(x-e2, u+e1, x_prev) - self.cost(x+e2, u-e1, x_prev) + self.cost(x-e2, u-e1, x_prev)
                c_ux[a, b] = c_ux[a, b]/(4*h**2)
        
        return c_x, c_u, c_xx, c_uu, c_ux
        
    def forward_pass(self, x0, k, K, x, u):
        x_trajectory = np.zeros((self.n_state, self.horizon+1))
        u_trajectory = np.zeros((self.n_control, self.horizon))
        
        x_trajectory[:, 0] = x0
        trajectory_cost = 0
        
        for t in range(self.horizon):
            u_trajectory[:, t] = u[:,t] + self.alpha * k[:, t] + K[:, :, t] @ (x_trajectory[:, t] - x[:, t])
            x_trajectory[:, t+1] = self.dynamics(x_trajectory[:, t], u_trajectory[:, t], t)
            trajectory_cost += self.cost(x_trajectory[:, t+1], u_trajectory[:, t], x_trajectory[:,t])
        
        trajectory_cost += self.terminal_cost(x_trajectory[:, -1])
        
        return x_trajectory, u_trajectory, trajectory_cost
        
    def backward_pass(self, x_trajectory, u_trajectory):
        V_x = np.zeros((self.n_state, self.horizon+1))
        V_xx = np.zeros((self.n_state, self.n_state, self.horizon+1))
        A_t = np.zeros((self.n_state, self.n_state, self.horizon))
        B_t = np.zeros((self.n_state, self.n_control, self.horizon))
        c_x_t = np.zeros((self.n_state, self.horizon))
        c_u_t = np.zeros((self.n_control, self.horizon))
        c_xx_t = np.zeros((self.n_state, self.n_state, self.horizon))
        c_uu_t = np.zeros((self.n_control, self.n_control, self.horizon))
        c_ux_t = np.zeros((self.n_control, self.n_state, self.horizon))
        k = np.zeros((self.n_control, self.horizon))
        K = np.zeros((self.n_control, self.n_state, self.horizon))
        
        V_x[:, -1], V_xx[:, :, -1] = self.quadratize_cost(x_trajectory[:, -1].flatten(), x_prev=x_trajectory[:,-2], index=self.horizon, u=None)
        
        for i in range(self.horizon):
            A_t[:, :, i], B_t[:, :, i] = self.linearize_dynamics(x_trajectory[:, i], x_trajectory[:,i+1], u_trajectory[:, i], i)
            
        for i in range(self.horizon):
            c_x_t[:, i], c_u_t[:, i], c_xx_t[:, :, i], c_uu_t[:, :, i], c_ux_t[:, :, i] = self.quadratize_cost(x_trajectory[:, i+1], u_trajectory[:, i], x_trajectory[:,i], i)
           
        delta_J1 = 0
        delta_J2 = 0
        for i in reversed(range(self.horizon)):
            Qx = c_x_t[:, i] + A_t[:, :, i].T @ V_x[:, i+1]
            Qu = c_u_t[:, i] + B_t[:, :, i].T @ V_x[:, i+1]
            Qxx = c_xx_t[:, :, i] + A_t[:, :, i].T @ V_xx[:, :, i+1] @ A_t[:, :, i]
            Quu = c_uu_t[:, :, i] + B_t[:, :, i].T @ V_xx[:, :, i+1] @ B_t[:, :, i]
            Qux = c_ux_t[:, :, i] + B_t[:, :, i].T @ V_xx[:, :, i+1] @ A_t[:, :, i]
            
            eig_Quu = np.linalg.eigvals(Quu)
            if np.min(eig_Quu) < 0:
                Quu = Quu - 2*np.min(eig_Quu)*np.eye(self.n_control)
            Quu_inv = np.linalg.pinv(Quu)
            
            k[:, i] = -Quu_inv @ Qu
            K[:, :, i] = -Quu_inv @ Qux
            
            V_x[:, i] = Qx + K[:,:,i].T @ Quu @ k[:,i] + K[:,:,i].T @ Qu + Qux.T @ k[:,i]
            V_xx[:, :, i] = Qxx + K[:,:,i].T @ Quu @ K[:,:,i] + K[:,:,i].T @ Qux + Qux.T @ K[:,:,i]
            
            delta_J1 = delta_J1 + self.alpha * k[:, i].T @ Qu
            delta_J2 = delta_J2 + (self.alpha**2/2) * k[:, i].T @ Quu @ k[:, i]
        
        self.delta_J = delta_J1 + delta_J2
        
        return k, K, V_x, V_xx
            
    def update_alpha(self):
        if self.alpha > self.alpha_floor:
            self.delta_J = self.delta_J / (self.alpha - (self.alpha**2)/2)
            self.alpha = 0.9 * self.alpha
            self.delta_J = self.delta_J * (self.alpha - (self.alpha**2)/2)
        else:
            if self.verbose:
                print('Minimum value of alpha reached:', self.alpha)
            
    def main_func(self, u_init, x0):
        if u_init is None:
            u_trajectory = np.zeros((self.n_control, self.horizon))
        else:
            u_trajectory = u_init.copy()
        k = np.zeros((self.n_control, self.horizon))
        K = np.zeros((self.n_control, self.n_state, self.horizon))
        
        x_trajectory, u_trajectory, cost = self.forward_pass(x0, k, K, np.zeros((self.n_state, self.horizon+1)), u_init)
        costs = [cost]
        print(f"Iteration {0}: Cost = {cost.item():.6f}, Alpha = {self.alpha:.6e}")
        for iteration in range(self.max_iterations):
            k, K, V_x, V_xx = self.backward_pass(x_trajectory, u_trajectory)
            flag = True
            while flag:  # Line search loop
                x_new, u_new, cost_new = self.forward_pass(x0, k, K, x_trajectory, u_trajectory)
                if cost_new <= cost:
                    flag = False
                    if np.abs((cost - cost_new)/cost) < 1e-4:
                        return x_trajectory, u_trajectory, np.array(costs)
                    x_trajectory, u_trajectory, cost = x_new, u_new, cost_new
                    costs.append(cost)
                    if self.verbose:
                        print(f"Iteration {iteration+1}: Cost = {cost.item():.6f}, Alpha = {self.alpha:.6e}")
                        print(f"Final State={x_new[:,-1]}")
                else:
                    self.update_alpha()  # Decay alpha
                    if self.alpha <= self.alpha_floor:
                        if self.verbose:
                            print(f"Iteration {iteration+1}: Line search failed, cost = {cost.item():.6f}")
                        return x_trajectory, u_trajectory, np.array(costs)
            
        return x_trajectory, u_trajectory, np.array(costs)


