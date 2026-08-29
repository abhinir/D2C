import numpy as np
import matplotlib.pyplot as plt

class ModelFree_ILQR:

    def __init__(self,
        model: callable,
        max_iterations: int = 100,
        alpha: float = 1.0,
        alpha_floor: float = 1e-8,
        std_dev_per: float = 1e-12,
        num_rollouts: int = 500,
        slid_window_per: int = 100,
        regularization: str = "Levenberg-Marquadt",
        verbose: bool = True):
        
        self.model = model
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.alpha = alpha
        self.alpha_floor = alpha_floor
        self.lr = []
        
        self.x_optimal = np.zeros((self.model.nx, self.model.horizon+1))
        self.u_optimal = np.zeros((self.model.nu, self.model.horizon))
        
        self.k = np.zeros((self.model.nu, self.model.horizon))
        self.K = np.zeros((self.model.nu, self.model.nx, self.model.horizon))
        self.Vx = np.zeros((self.model.nx, self.model.horizon+1))
        self.Vxx = np.zeros((self.model.nx, self.model.nx, self.model.horizon+1))
        
        self.costs = []
        self.delta_J = []
        self.delta_J_curr = 0.0
        self.cost_change_ratio = []
        
        self.regularization = regularization
        self.mu = 1e-3
        self.mu_min = 1e-8
        self.mu_max = 1e8
        self.delta_0 = 2
        self.delta = self.delta_0
        
        self.eps = 1e-5
        
        self.std_dev_per = std_dev_per
        self.num_rollouts = num_rollouts
        self.slid_window_per = slid_window_per
        
        
#    def linearize_dynamics(self, x, x_next, u, index):
#        
#        dx = np.zeros((self.n_state, self.num_rollouts))
#        du = np.zeros((self.n_control, self.num_rollouts))
#        dx_next = np.zeros((self.n_state, self.num_rollouts))
#        x_per = np.zeros((self.n_state, 2))
#        u_per = np.zeros((self.n_control))
#        for i in range(self.num_rollouts):
#            dx[:,i] = np.random.normal(loc=0.0, scale = self.std_dev_per, size = self.n_state)
#            du[:,i] = np.random.normal(loc=0.0, scale = self.std_dev_per, size = self.n_control)
#            dx_next[:,i] = self.dynamics(x + dx[:,i], u + du[:,i], index) - x_next
#            
#        
#        Y = dx_next
#        X = np.vstack((dx, du))
#        P = Y @ np.linalg.pinv(X)
#        A = P[:, :self.n_state]
#        B = P[:, self.n_state:]
#        
#        
#        
#        return A, B
        
        
    def linearize_dynamics(self, x, x_alt, u):
            x_per = np.repeat(x[:,:,None],self.num_rollouts, axis = 2)
            x_alt_per = np.repeat(x_alt[:,:,None],self.num_rollouts, axis = 2)
            u_per = np.repeat(u[:,:,None],self.num_rollouts, axis = 2)
            t = 0
            for p in range(self.num_rollouts):
                for t in range(self.model.horizon):
                    u_per[:,t,p] += np.random.normal(loc=0.0, scale = self.std_dev_per)
                    x_per[:,t+1,p], x_alt_per[:,t+1,p] = self.model.propagate_dynamics(x_per[:,t,p], u_per[:,t,p], t)
                    #x_alt_per[:,t+1,p] = self.alternate_state(x_per[:,t+1,p])
                        
                
            A = np.zeros((self.model.nx, self.model.nx, self.model.horizon))
            J_alt = np.zeros((self.model.nx, self.model.nx_alt, self.model.horizon))
            B = np.zeros((self.model.nx, self.model.nu, self.model.horizon))
            
            for i in range(self.model.horizon):
                dx = (x_per[:,i,:].T - x[:,i]).T
                dx_alt = (x_alt_per[:,i,:] - x_alt[x,:]).T
                du = (u_per[:,i,:].T - u[:,i]).T
                dx_next = (x_per[:,i+1,:].T - x[:,i+1]).T
                X = np.vstack((dx, dx_alt, du))
                P = dx_next @ np.linalg.pinv(X)
                A[:,:,i] = P[:, :self.model.nx]
                J_alt[:,:,i] = P[:, self.model.nx:(self.model.nx + self.model.nx_alt)]
                B[:,:,i] = P[:, (self.model.nx + self.model.nx_alt):]
                #J_alt[:,:,i] = 
            
            return A, J_alt, B

    def quadratize_cost(self, x, u, index):
        if u is None or index == self.model.horizon:
            c_x = np.zeros((self.model.nx))
            c_xx = np.zeros((self.model.nx, self.model.nx))
            h = self.eps
            for i in range(self.model.nx):
                e = np.zeros((self.model.nx))
                e[i] = h
                c_x[i] = (self.model.terminal_cost(x + e) - self.model.terminal_cost(x - e))/(2*h)
            for a in range(self.model.nx):
                for b in range(self.model.nx):
                    e1 = np.zeros((self.model.nx))
                    e1[a] = h
                    e2 = np.zeros((self.model.nx))
                    e2[b] = h
                    c_xx[a, b] = self.model.terminal_cost(x+e1+e2) - self.model.terminal_cost(x+e1-e2) - self.model.terminal_cost(x-e1+e2) + self.model.terminal_cost(x-e1-e2)
                    c_xx[a, b] = c_xx[a, b]/(4*h*h)
            return c_x, c_xx
            
        c_x = np.zeros((self.model.nx))
        c_u = np.zeros((self.model.nu))
        h = self.eps
        for i in range(self.model.nx):
            e = np.zeros((self.model.nx))
            e[i] = h
            c_x[i] = (self.model.cost(x + e, u) - self.model.cost(x - e, u))/(2*h)
        
        for j in range(self.model.nu):
            e = np.zeros((self.model.nu))
            e[j] = h
            c_u[j] = (self.model.cost(x, u + e) - self.model.cost(x, u - e))/(2*h)
        
        c_xx = np.zeros((self.model.nx, self.model.nx))
        c_uu = np.zeros((self.model.nu, self.model.nu))
        c_ux = np.zeros((self.model.nu, self.model.nx))
        for a in range(self.model.nx):
            for b in range(self.model.nx):
                e1 = np.zeros((self.model.nx))
                e1[a] = h
                e2 = np.zeros((self.model.nx))
                e2[b] = h
                c_xx[a, b] = self.model.cost(x+e1+e2, u) - self.model.cost(x+e1-e2, u) - self.model.cost(x-e1+e2, u) + self.model.cost(x-e1-e2, u)
                c_xx[a, b] = c_xx[a, b]/(4*h*h)
        
        for a in range(self.model.nu):
            for b in range(self.model.nu):
                e1 = np.zeros((self.model.nu))
                e1[a] = h
                e2 = np.zeros((self.model.nu))
                e2[b] = h
                c_uu[a, b] = self.model.cost(x, u+e1+e2) - self.model.cost(x, u+e1-e2) - self.model.cost(x, u-e1+e2) + self.model.cost(x, u-e1-e2)
                c_uu[a, b] = c_uu[a, b]/(4*h*h)
        
        for a in range(self.model.nu):
            for b in range(self.model.nx):
                e1 = np.zeros((self.model.nu))
                e1[a] = h
                e2 = np.zeros((self.model.nx))
                e2[b] = h
                c_ux[a, b] = self.model.cost(x+e2, u+e1) - self.model.cost(x-e2, u+e1) - self.model.cost(x+e2, u-e1) + self.model.cost(x-e2, u-e1)
                c_ux[a, b] = c_ux[a, b]/(4*h**2)
        
        return c_x, c_u, c_xx, c_uu, c_ux
    
    def inc_regularization(self):
        self.delta = np.maximum(self.delta_0, self.delta_0*self.delta)
        self.mu *= self.delta
        self.mu = self.mu_max if self.mu>self.mu_max else self.mu
    
    def dec_regularization(self):
        self.delta = np.minimum(1/self.delta_0, self.delta/self.delta_0)
        self.mu *= self.delta
        self.mu = self.mu_min if self.mu < self.mu_min else self.mu
        
    def forward_pass(self, x0, x, u):
        x_trajectory = np.zeros((self.model.nx, self.model.horizon+1))
        x_alt_trajectory = np.zeros((self.model.nx_alt, self.model.horizon+1))
        u_trajectory = np.zeros((self.model.nu, self.model.horizon))
        
        x_trajectory[:, 0] = x0
        trajectory_cost = 0
        
        for t in range(self.model.horizon):
            u_trajectory[:, t] = u[:,t] + self.alpha * self.k[:, t] + self.K[:, :, t] @ (x_trajectory[:, t] - x[:, t])
            x_trajectory[:, t+1], x_alt_trajectory[:, t+1] = self.model.propagate_dynamics(x_trajectory[:, t], u_trajectory[:, t], t)
            trajectory_cost += self.model.cost(x_trajectory[:, t], x_alt_trajectory[:, t], u_trajectory[:, t])
            
        trajectory_cost += self.model.terminal_cost(x_trajectory[:, -1], x_alt_trajectory[:, -1])
        
        return x_trajectory, x_alt_trajectory, u_trajectory, trajectory_cost
        
    def backward_pass(self, x_trajectory, x_alt_trajectory, u_trajectory):
        A_t = np.zeros((self.model.nx, self.model.nx, self.model.horizon))
        J_alt_t = np.zeros((self.model.nx, self.model.nx_alt, self.model.horizon))
        B_t = np.zeros((self.model.nx, self.model.nu, self.model.horizon))
        c_x_t = np.zeros((self.model.nx, self.model.horizon))
        c_u_t = np.zeros((self.model.nu, self.model.horizon))
        c_xx_t = np.zeros((self.model.nx, self.model.nx, self.model.horizon))
        c_uu_t = np.zeros((self.model.nu, self.model.nu, self.model.horizon))
        c_ux_t = np.zeros((self.model.nu, self.model.nx, self.model.horizon))
        
        self.Vx[:, -1], self.Vxx[:, :, -1] = self.quadratize_cost(x_alt_trajectory[:, -1].flatten(), index=self.model.horizon, u=None)
        
#        for i in range(self.horizon):
#            A_t[:,:,i], B_t[:,:,i] = self.linearize_dynamics(x_trajectory[:,i], x_trajectory[:,i+1], u_trajectory[:,i], i)
            
        A_t, J_alt_t, B_t = self.linearize_dynamics(x_trajectory, x_alt_trajectory, u_trajectory)
        
        
        for i in range(self.model.horizon):
            c_x_t[:, i], c_u_t[:, i], c_xx_t[:, :, i], c_uu_t[:, :, i], c_ux_t[:, :, i] = self.quadratize_cost(x_trajectory[:, i], u_trajectory[:, i], i)
#        for i in range(self.horizon):
#            c_x_t[:, i] = self.Q @ x_trajectory[:, i]
#            c_u_t[:, i] = self.R @ u_trajectory[:, i]
#            c_xx_t[:, :, i] = self.Q
#            c_uu_t[:, :, i] = self.R
#            c_ux_t[:, :, i] = np.zeros((self.n_control, self.n_state))
           
        delta_J1 = 0
        delta_J2 = 0
        for i in reversed(range(self.model.horizon)):
            Qx = J_alt_t[:, :, i] @ c_x_t[:, i] + A_t[:, :, i].T @ self.Vx[:, i+1]
            Qu = c_u_t[:, i] + B_t[:, :, i].T @ self.Vx[:, i+1]
            Qxx = J_alt_t[:, :, i] @ c_xx_t[:, :, i] @ J_alt_t[:, :, i].T + A_t[:, :, i].T @ self.Vxx[:, :, i+1] @ A_t[:, :, i]
            Quu = c_uu_t[:, :, i] + B_t[:, :, i].T @ self.Vxx[:, :, i+1] @ B_t[:, :, i]
            Qux = c_ux_t[:, :, i] @ J_alt_t.T[:, :, i] + B_t[:, :, i].T @ self.Vxx[:, :, i+1] @ A_t[:, :, i]
            
            Quu = 0.5 * (Quu + Quu.T)
            Qxx = 0.5 * (Qxx + Qxx.T)
            eig_Quu = np.linalg.eigvals(Quu)
#            regularization = 1e6
            if np.min(eig_Quu) < 1e-6:
                Quu_reg = Quu - np.min(eig_Quu)*np.eye(self.model.nu) + 1000*np.eye(self.model.nu)
            #if np.min(eig_Quu) <= 1e-6:
            #    print(f"Quu is not positive definite.Min Eigenvalue:{np.min(eig_Quu)}")
            #    regularization = -np.min(eig_Quu)+10
            #    Quu_reg = Quu + regularization*np.eye(self.model.nu)
            #    eig_Quu = np.linalg.eigvals(Quu_reg)
            else:
                Quu_reg = Quu
                
            Quu_inv = np.linalg.pinv(Quu_reg)
            
            
            self.k[:, i] = -Quu_inv @ Qu
            self.K[:, :, i] = -Quu_inv @ Qux
            
            
            
            self.Vx[:, i] = Qx + self.K[:,:,i].T @ Quu @ self.k[:,i] + self.K[:,:,i].T @ Qu + Qux.T @ self.k[:,i]
            self.Vxx[:, :, i] = Qxx + self.K[:,:,i].T @ Quu @ self.K[:,:,i] + self.K[:,:,i].T @ Qux + Qux.T @ self.K[:,:,i]
            
            delta_J1 = delta_J1 + self.alpha * self.k[:, i].T @ Qu
            delta_J2 = delta_J2 + (self.alpha**2/2) * self.k[:, i].T @ Quu @ self.k[:, i]
        
        self.delta_J_curr = delta_J1 + delta_J2
    
    def backward_pass2(self, x_trajectory, u_trajectory):
        A_t = np.zeros((self.model.nx, self.model.nx, self.model.horizon))
        B_t = np.zeros((self.model.nx, self.model.nu, self.model.horizon))
        c_x_t = np.zeros((self.model.nx, self.model.horizon))
        c_u_t = np.zeros((self.model.nu, self.model.horizon))
        c_xx_t = np.zeros((self.model.nx, self.model.nx, self.model.horizon))
        c_uu_t = np.zeros((self.model.nu, self.model.nu, self.model.horizon))
        c_ux_t = np.zeros((self.model.nu, self.model.nx, self.model.horizon))
        
        self.Vx[:, -1], self.Vxx[:, :, -1] = self.quadratize_cost(x_trajectory[:, -1].flatten(), index=self.model.horizon, u=None)
        
#        for i in range(self.horizon):
#            A_t[:,:,i], B_t[:,:,i] = self.linearize_dynamics(x_trajectory[:,i], x_trajectory[:,i+1], u_trajectory[:,i], i)
            
        A_t, B_t = self.linearize_dynamics(x_trajectory,u_trajectory)
        
        
        for i in range(self.model.horizon):
            c_x_t[:, i], c_u_t[:, i], c_xx_t[:, :, i], c_uu_t[:, :, i], c_ux_t[:, :, i] = self.quadratize_cost(x_trajectory[:, i], u_trajectory[:, i], i)
#        for i in range(self.horizon):
#            c_x_t[:, i] = self.model.Q @ (x_trajectory[:, i] - self.model.xg)
#            c_u_t[:, i] = self.model.R @ u_trajectory[:, i]
#            c_xx_t[:, :, i] = self.model.Q
#            c_uu_t[:, :, i] = self.model.R
#            c_ux_t[:, :, i] = np.zeros((self.n_control, self.n_state))
           
        delta_J1 = 0
        delta_J2 = 0
        for i in reversed(range(self.model.horizon)):
            Qx = c_x_t[:, i] + A_t[:, :, i].T @ self.Vx[:, i+1]
            Qu = c_u_t[:, i] + B_t[:, :, i].T @ self.Vx[:, i+1]
            Qxx = c_xx_t[:, :, i] + A_t[:, :, i].T @ self.Vxx[:, :, i+1] @ A_t[:, :, i]
            Quu = c_uu_t[:, :, i] + B_t[:, :, i].T @ (self.Vxx[:, :, i+1] + self.mu*np.eye*(self.model.nx)) @ B_t[:, :, i]
            Qux = c_ux_t[:, :, i] + B_t[:, :, i].T @ (self.Vxx[:, :, i+1] + self.mu*np.eye*(self.model.nx)) @ A_t[:, :, i]
            
            Quu = 0.5 * (Quu + Quu.T)
            Qxx = 0.5 * (Qxx + Qxx.T)
            eig_Quu = np.linalg.eigvals(Quu)
#            regularization = 1e6
            if np.min(eig_Quu) < 0 and (self.mu < self.mu_max):
                self.inc_regularization()
                return True
            else:
                self.dec_regularization()
                
            Quu_inv = np.linalg.pinv(Quu)
            
            
            self.k[:, i] = -Quu_inv @ Qu
            self.K[:, :, i] = -Quu_inv @ Qux
            
            
            
            self.Vx[:, i] = Qx + self.K[:,:,i].T @ Quu @ self.k[:,i] + self.K[:,:,i].T @ Qu + Qux.T @ self.k[:,i]
            self.Vxx[:, :, i] = Qxx + self.K[:,:,i].T @ Quu @ self.K[:,:,i] + self.K[:,:,i].T @ Qux + Qux.T @ self.K[:,:,i]
            
            delta_J1 = delta_J1 + self.alpha * self.k[:, i].T @ Qu
            delta_J2 = delta_J2 + (self.alpha**2/2) * self.k[:, i].T @ Quu @ self.k[:, i]
        
        self.delta_J_curr = delta_J1 + delta_J2
        return False
        
            
    def update_alpha(self):
        if self.alpha > self.alpha_floor:
            self.delta_J_curr = self.delta_J_curr / (self.alpha - (self.alpha**2)/2)
            self.alpha = 0.9 * self.alpha
            self.delta_J_curr = self.delta_J_curr * (self.alpha - (self.alpha**2)/2)
        else:
            if self.verbose:
                print('Minimum value of alpha reached:', self.alpha)
    
    def check_convergence(self):
        if np.linalg.norm(self.k.T, np.inf) < 1e-4:
            print("Optimality:",np.linalg.norm(self.k.T, np.inf))
            convergence_criteria = True
            if self.verbose:
                print('Optimality Satisfied.')
        elif abs(self.costs[-1] - self.costs[-2])/abs(self.costs[-1]) < 1e-4:
            convergence_criteria = True
            if self.verbose:
                print(f"Change in cost is very small: {abs(self.costs[-1] - self.costs[-2])*100/abs(self.costs[-1])}%.")
        else:
            print("Optimality:",np.linalg.norm(self.k.T, np.inf))
            convergence_criteria = False
        
        conv3 = 1
        if len(self.costs) >3:
            conv3 = 0
            for i in range(3):
                conv3 = conv3 + abs(self.costs[-1 - i] - self.costs[-2 - i])/abs(self.costs[-1 - i])
        
        if conv3 < 1e-2:
            convergence_criteria = True
            if self.verbose:
                print(f"Change in cost over past three iterations is small: {conv3*100}%.")
        
        return convergence_criteria
        
        
    def main_func(self, u_init, x0):
        if u_init is None:
            u_trajectory = np.zeros((self.model.nu, self.model.horizon))
        else:
            u_trajectory = u_init.copy()
        
        
        x_trajectory, x_alt_trajectory, u_trajectory, cost = self.forward_pass(x0, np.zeros((self.model.nx, self.model.horizon+1)), u_init)
        self.costs.append(cost.item())
        print(f"Iteration {0}: Cost = {cost.item():.6f}, Alpha = {self.alpha:.6e}")
        for iteration in range(self.max_iterations):
            if self.regularization == "Levenberg-Marquadt":
                self.backward_pass(x_trajectory, x_alt_trajectory, u_trajectory)
            else:
                while backward_flag:
                    backward_flag = self.backward_pass2(x_trajectory, u_trajectory)
                
            forward_flag = True
            while forward_flag:  # Line search loop
                x_new, x_alt_new, u_new, cost_new = self.forward_pass(x0, x_trajectory, u_trajectory)
                if cost_new <= cost:
                    forward_flag = False
                    x_trajectory, x_alt_trajectory, u_trajectory, cost = x_new, x_alt_new, u_new, cost_new
                    self.costs.append(cost.item())
                    self.delta_J.append(self.delta_J_curr)
                    self.lr.append(self.alpha)
                    if self.verbose:
                        print(f"Iteration {iteration+1}: Cost = {cost.item():.6f}, Alpha = {self.alpha:.6e}")
                        print(f"Final (Alternate) State = {x_alt_trajectory[:,-1]}")
                    if self.check_convergence():
                        self.x_optimal = x_new
                        self.x_alt_optimal = x_alt_new
                        self.u_optimal = u_new
                        self.costs = np.array(self.costs)
                        self.delta_J = np.array(self.delta_J)
                        self.cost_change_ratio = np.array((self.costs[1:] - self.costs[:-1])/self.delta_J)
                        self.lr = np.array(self.lr)
                        return 0
                    
                else:
                    self.update_alpha()  # Decay alpha
                    if self.alpha <= self.alpha_floor:
                        self.x_optimal = x_new
                        self.x_alt_optimal = x_alt_new
                        self.u_optimal = u_new
                        self.costs = np.array(self.costs)
                        self.delta_J = np.array(self.delta_J)
                        self.cost_change_ratio = np.array((self.costs[1:] - self.costs[:-1])/self.delta_J)
                        self.lr = np.array(self.lr)
                        if self.verbose:
                            print(f"Iteration {iteration+1}: Line search failed, cost = {cost.item():.6f}")
                        return 0
            

