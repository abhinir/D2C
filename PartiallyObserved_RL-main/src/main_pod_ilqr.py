import numpy as np
import math
from main_ilqr import iLQR
from arma_ltv_sys_id import ARMA_LTV_SysID
np.random.seed(42)

class POD_iLQR(iLQR):

    def __init__(self, C, MODEL, n_x, n_u, alpha, horizon, init_state, final_state, n_z, q, q_u, Q, Q_final, R, 
                 nominal_init_stddev, n_sys_id_samples, pert_sys_id_sigma, arma_sys_id_flag = True):
        self.C = C
        self.n_z = n_z
        self.q = q
        self.q_u = q_u
        self.n_aug = n_z*q+n_u*(q_u-1)
        iLQR.__init__(self, MODEL, n_x, n_u, alpha, horizon, init_state, final_state, 
                      Q, Q_final, R, nominal_init_stddev, n_sys_id_samples, pert_sys_id_sigma, arma_sys_id_flag = arma_sys_id_flag)
        
        self.Z_aug_0 = np.zeros((self.n_aug,1))
        self.Z_aug_0[0:n_z,:] = self.C @ self.X_0
        self.Z_aug = np.zeros((self.N, self.n_aug, 1))

        self.Z = np.zeros((self.N, self.n_z, 1))
        self.Z_temp = np.zeros((self.N, self.n_z, 1))

        self.K = np.zeros((self.N, self.n_u, self.n_aug))
        self.k = np.zeros((self.N, self.n_u, 1))
		
        self.V_xx = np.zeros((self.N, self.n_aug, self.n_aug))
        self.V_x = np.zeros((self.N, self.n_aug, 1))
        
        self.ltv_sys_id = ARMA_LTV_SysID(self.model, self.n_x, n_u, n_z, C, q, q_u, self.N, n_samples=n_sys_id_samples, pert_sigma = pert_sys_id_sigma)
        
    def iterate_ilqr(self, n_iter, u_init=None):
        # exactly same from iLQR, will be removed later
        '''
			Main function that carries out the algorithm at higher level
            n_iter : number of iLQR iterations
		'''

		# Initialize the trajectory with the desired initial guess
        self.initialize_traj(u_init=u_init)

        # Start the iLQR iterations
        for i in self.pbar(range(n_iter)):
            backward_pass_flag, del_J_alpha = self.backward_pass()

            if backward_pass_flag:
                self.dec_reg_mu()
                forward_pass_flag = self.forward_pass(del_J_alpha)

                if not forward_pass_flag:
                    while not forward_pass_flag:
                        # simulated annealing
                        self.alpha = self.alpha*0.99
                        forward_pass_flag = self.forward_pass(del_J_alpha)
            else:
                self.inc_reg_mu()
                print(f"Iteration {i} failed.")

            if i<5:
                self.alpha = self.alpha*0.9
            else:
                self.alpha = self.alpha*0.999

            self.episodic_cost_history.append(self.calculate_total_cost(self.Z_aug_0, self.Z_aug, self.U, self.N))

    def backward_pass(self):
        """
        Carry out the backward pass to compute the feedforward and feedback gains
        returns : backward_pass_flag : indicates if backward pass was successful
                  del_J_alpha : expected cost reduction
        """
        ################## defining local functions & variables for faster access ################
        k = np.copy(self.k)
        K = np.copy(self.K)
        V_x = np.copy(self.V_x)
        V_xx = np.copy(self.V_xx)
        ##########################################################################################

        V_x[self.N-1] = self.l_x_N(self.Z_aug[self.N-1])	
        V_xx[self.N-1] = 2*self.Q_final

        # Initialize before forward pass
        del_J_alpha = 0

        Fx_Fu = self.ltv_sys_id.traj_sys_id(np.concatenate((self.X_0.reshape(1, self.n_x, 1), self.X), axis=0), self.U) # only init state and Z trajectory needed
        
        for t in range(self.N-1, max(self.q, self.q_u)-1, -1):
            F_x = Fx_Fu[t][:,:self.n_aug]
            F_u = Fx_Fu[t][:,self.n_aug:]

            if t>0:
                Q_x, Q_u, Q_xx, Q_uu, Q_ux = self.get_gradients(F_x,F_u,self.Z_aug[t-1],self.U[t],V_x[t], V_xx[t])
            else:
                Q_x, Q_u, Q_xx, Q_uu, Q_ux = self.get_gradients(F_x,F_u,self.Z_aug_0,self.U[0],V_x[0], V_xx[0])
            
            
            try:
                np.linalg.cholesky(Q_uu)

            except np.linalg.LinAlgError:
                print("FAILED! Q_uu is not Positive definite at t=",t)
                backward_pass_flag = 0
                k = np.copy(self.k)
                K = np.copy(self.K)
                V_x = np.copy(self.V_x)
                V_xx = np.copy(self.V_xx)
                break

            else:
                backward_pass_flag = 1
                # update gains as follows
                Q_uu_inv = np.linalg.inv(Q_uu)
                k[t] = -(Q_uu_inv @ Q_u)
                K[t] = -(Q_uu_inv @ Q_ux)

                del_J_alpha += -self.alpha*((k[t].T) @ Q_u) - 0.5*self.alpha**2 * ((k[t].T) @ (Q_uu @ k[t]))
				
                if t>0:
                    V_x[t-1] = Q_x + (K[t].T) @ (Q_uu @ k[t]) + ((K[t].T) @ Q_u) + ((Q_ux.T) @ k[t])
                    V_xx[t-1] = Q_xx + ((K[t].T) @ (Q_uu @ K[t])) + ((K[t].T) @ Q_ux) + ((Q_ux.T) @ K[t])

		######################### Update the new gains ##############################################
        self.k = np.copy(k)
        self.K = np.copy(K)
        self.V_x = np.copy(V_x)
        self.V_xx = np.copy(V_xx)

        return backward_pass_flag, del_J_alpha
    
    def forward_pass(self, del_J_alpha):
        """
            Forward pass with line search
            del_J_alpha : expected cost reduction scaled with alpha
            returns : forward_pass_flag : 1 if forward pass is successful else 0
        """
        #cost before forward pass
        J1 = self.calculate_total_cost(self.Z_aug_0, self.Z_aug, self.U, self.N)

        self.X_temp = np.copy(self.X)
        self.Z_temp = np.copy(self.Z)
        self.U_temp = np.copy(self.U)

        self.forward_pass_simulate()

        #cost after forward pass
        J2 = self.calculate_total_cost(self.Z_aug_0, self.Z_aug, self.U, self.N)

        if (J1-J2)/del_J_alpha < self.J_change_eps:
            forward_pass_flag = 0
            self.X = np.copy(self.X_temp)
            self.Z = np.copy(self.Z_temp)
            self.U = np.copy(self.U_temp)
        else:
            forward_pass_flag = 1

        return forward_pass_flag
    
    def forward_pass_simulate(self):
        """ 
        Simulate the system with updated controls 
        """
        ################## defining local functions & variables for faster access ################
        n_z, n_u, q, q_u = self.n_z, self.n_u, self.q, self.q_u
		##########################################################################################
        z = np.zeros((n_z*q,1))
        u = np.zeros((n_u*q_u,1))
        d_z = np.zeros((n_z*q,1))
        d_u = np.zeros((n_u*q_u,1))

        for t in range(self.N):
            if t < max(q, q_u):
                self.U[t] = self.U_temp[t] + self.alpha*self.k[t]
                d_u[n_u*(q_u-t-1):n_u*(q_u-t)] = self.U[t]-self.U_temp[t]
                u[n_u*(q_u-t-1):n_u*(q_u-t)] = self.U[t]
                if t != 0:
                    d_z[n_z*(q-t-1):n_z*(q-t)] = self.Z[t-1] - self.Z_temp[t-1]	
                    z[n_z*(q-t-1):n_z*(q-t)] = self.Z[t-1]
            else:
                d_z_prev = d_z[:n_z*(q-1)]
                d_z[n_z:] = d_z_prev
                z_prev = z[:n_z*(q-1)]
                z[n_z:] = z_prev

                d_u_prev = d_u[:n_u*(q_u-1)]
                d_u[n_u:] = d_u_prev
                u_prev = u[:n_u*(q_u-1)]
                u[n_u:] = u_prev

                d_z[:n_z] = self.Z[t-1] - self.Z_temp[t-1]
                z[:n_z] = self.Z[t-1]

                self.U[t] = self.U_temp[t] + self.alpha*self.k[t] + (self.K[t] @ np.vstack([d_z, d_u[n_u:]]))
                d_u[:n_u] = self.U[t]-self.U_temp[t]
                u[:n_u] = self.U[t]
                self.Z_aug[t] = np.vstack([z, u[n_u:]])
                # self.Z_aug[t] = np.vstack([d_z, d_u[n_u:]])



            if t==0:
                self.X[t] = self.model.simulate_step(self.X_0.flatten(),self.U[t].flatten()).reshape(np.shape(self.X_0))
                self.Z[t] = self.C @ self.X[t]
            else:
                self.X[t] = self.model.simulate_step(self.X[t-1].flatten(),self.U[t].flatten()).reshape(np.shape(self.X_0))
                self.Z[t] = self.C @ self.X[t]

    def initialize_traj(self,u_init):
        """
        Initialize the nominal trajectory with an initial guess for control
        u_init : (N, n_u, 1)
        """
        if u_init is None:
            self.U = np.random.normal(0, self.nominal_init_stddev, (self.N, self.n_u, 1))
        else:
            self.U = u_init #TODO: check the shape of u_init

        self.U_temp = self.U
        self.forward_pass_simulate()
        self.X_temp = self.X
        self.Z_temp = self.Z

    # def calculate_total_cost(self,X_0, state_traj, control_traj, horizon):
    #     total_cost = 0
    #     total_cost += self.incremental_cost(X_0,control_traj[0])
    #     for t in range(horizon-1):
    #         total_cost += self.incremental_cost(state_traj[t],control_traj[t+1])
    #     total_cost += self.terminal_cost(state_traj[horizon-1])

    #     return total_cost
    
    # def incremental_cost(self,x,u):
    #     '''
	# 		Incremental cost in terms of state and controls.
    #         Can be overwritten in actual working example
	# 	'''
    #     return (((x - self.X_N).T @ self.Q) @ (x - self.X_N)) + (((u.T) @ self.R) @ u)
	
    # def terminal_cost(self,x):
    #     '''
	# 		Terminal cost in terms of state.
    #         Can be overwritten in actual working example
	# 	'''
    #     return (((x - self.X_N).T @ self.Q_final) @ (x - self.X_N)) 
    
    # def l_x(self, x):
    #     """
    #     Compute the gradient of the running cost
    #     x : (n_x,1)
    #     returns : l_x : (n_x,1)
    #     """
    #     z = np.zeros((np.shape(self.Q)[0], 1))
    #     z[:self.n_z,:] = self.C @(x - self.X_N)
    #     return 2*self.Q @ z

    # def l_x_N(self, x):
    #     """
    #     Compute the gradient of the terminal cost
    #     x : (n_x,1)
    #     returns : l_x_N : (n_x,1)
    #     """
    #     z = np.zeros((np.shape(self.Q)[0], 1))
    #     z[:self.n_z,:] = self.C @(x - self.X_N)
    #     return 2*self.Q_final @ z