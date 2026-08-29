import numpy as np
from ltv_sys_id import LTV_SysID
class ARMA_LTV_SysID(LTV_SysID):

    def __init__(self, MODEL, n_x, n_u, n_z, C, q, q_u, N, n_samples=500, pert_sigma = 1e-3):
        """
        ARMA LTV System Identification
        Args:
            MODEL: dynamics model
            n_x: full state dimension
            n_u: control dimension  
            n_z: observation dimension (e.g., 1 for pendulum, 2 for cartpole)
            q: state history length
            q_u: control history length 
            n_samples: number of perturbed trajectories
            pert_sigma: perturbation standard deviation for actions
        """
        # super().__init__(MODEL, n_x, n_u, N, n_samples, pert_sigma)
        LTV_SysID.__init__(self, MODEL, n_x, n_u, N, n_samples = n_samples, pert_sigma = pert_sigma)
        self.n_z = n_z
        self.C = C
        self.q = q
        self.q_u = q_u

    def traj_sys_id(self, x_nom, u_nom):
        '''
            system identification for a given nominal state and control
            x_nom = (N+1, n_x, 1)
            u_nom = (N, n_u, 1)
            returns - a numpy array with F_x and F_u horizantally stacked
		'''
		################## defining local functions & variables for faster access ################
        n_z, N = self.n_z, self.N
		##########################################################################################
        # Generating perturbations
        X_pertb, U_pertb = self.generate_rollouts(x_nom, u_nom)
        Z_nom = self.C @ x_nom
        Z = self.C @ X_pertb #TODO : (N+1,nx,n_samples) -> (N+1,nz,n_samples)
        
        # Generating delta_z for all rollouts
        delta_Z = np.zeros((N+1, n_z, self.n_samples))
        for i in range(N+1):
            delta_Z[i, :, :] = Z[i, :, :] - Z_nom[i, :, 0:1]
        
        delta_Z = delta_Z.transpose(2, 1, 0)  # (n_samples, n_z, N+1)
        U_pertb = U_pertb.transpose(2, 1, 0)  # (n_samples, n_u, N+1)
        
        return self.arma_fit(delta_Z, U_pertb)
    

    def arma_fit(self, delta_Z, U_pertb):
        """
        ARMA LTV fitting with forward time indexing
        delta_Z : (n_samples, n_z, N+1)
        U_pertb : (n_samples, n_u, N+1)
        returns : AB_aug : (N, aug_dim, aug_dim + n_u)
        """
        n_z, n_u, q, q_u, N, n_samples = self.n_z, self.n_u, self.q, self.q_u, self.N, self.n_samples
        
        # Augmented state dimension
        aug_dim = n_z*q + n_u*max(0, q_u-1)
        
        # Output arrays
        A_aug = np.zeros((N, aug_dim, aug_dim))
        B_aug = np.zeros((N, aug_dim, n_u))
        
        # Pre-allocate regressor arrays
        M = np.zeros((n_samples, n_z*q + n_u*q_u))
        target = np.zeros((n_samples, n_z))
        
        # Pre-compute constant shifting blocks
        state_shift_block = None
        if q > 1:
            state_eye = np.eye(n_z*(q-1))
            state_zeros = np.zeros((n_z*(q-1), n_z + n_u*max(0, q_u-1)))
            state_shift_block = np.hstack([state_eye, state_zeros])
        
        ctrl_eye = None
        if q_u > 2:
            ctrl_eye = np.eye(n_u*(q_u-2))
        
        b_state_zeros = np.zeros((n_z*max(0, q-1), n_u))
        
        if q_u > 1:
            b_ctrl_eye = np.eye(n_u)
            if q_u > 2:
                b_ctrl_zeros = np.zeros((n_u*(q_u-2), n_u))
                b_ctrl_block = np.vstack([b_ctrl_eye, b_ctrl_zeros])
            else:
                b_ctrl_block = b_ctrl_eye
        else:
            b_ctrl_block = np.zeros((0, n_u))
        
        # Main loop - forward in time
        start_idx = max(q, q_u)
        
        for t in range(start_idx, N+1):
            # Build regressor: [δz[t-q], ..., δz[t-1], δu[t-q_u], ..., δu[t-1]]
            for i in range(q):
                M[:, i*n_z:(i+1)*n_z] = delta_Z[:, :, t-q+i]
            
            for i in range(q_u):
                M[:, q*n_z + i*n_u : q*n_z + (i+1)*n_u] = U_pertb[:, :, t-q_u+i]
            
            # Target: δz[t]
            target[:, :] = delta_Z[:, :, t]
            
            # Solve least squares
            fitcoef, _, _, _ = np.linalg.lstsq(M, target, rcond=None)
            fitcoef = fitcoef.T  # (n_z, n_z*q + n_u*q_u)
            
            # Store at output index (t-1)
            out_idx = t - 1
            
            # === A_aug construction ===
            # Top row: [α_{t-1,1}, ..., α_{t-1,q}, β_{t-1,2}, ..., β_{t-1,q_u}]
            if q_u > 1:
                # Include state history and control history (excluding current control)
                A_aug[out_idx, :n_z, :n_z*q] = fitcoef[:, :n_z*q]
                A_aug[out_idx, :n_z, n_z*q:] = fitcoef[:, n_z*q+n_u:]
            else:
                # q_u=1: only state history (no control history to store)
                A_aug[out_idx, :n_z, :n_z*q] = fitcoef[:, :n_z*q]
            
            # State shifting block: [I_{n_z*(q-1)}, 0]
            if q > 1 and state_shift_block is not None:
                A_aug[out_idx, n_z:n_z*q, :] = state_shift_block
            
            # Control shifting block (only if q_u > 2)
            if q_u > 2 and ctrl_eye is not None:
                row_start = n_z*q + n_u
                col_start = n_z*q + n_u
                row_end = row_start + n_u*(q_u-2)
                col_end = col_start + n_u*(q_u-2)
                
                if row_end <= aug_dim and col_end <= aug_dim:
                    A_aug[out_idx, row_start:row_end, col_start:col_end] = ctrl_eye
            
            # === B_aug construction ===
            # Top part: [β_{t-1,1}; 0; ...; 0] (current control + zeros for state history)
            if q > 1:
                B_aug[out_idx, :n_z*q, :] = np.vstack([
                    fitcoef[:, n_z*q:n_z*q+n_u],  # Current control coefficient
                    b_state_zeros                  # Zeros for state history slots
                ])
            else:
                # q=1: just the current control coefficient
                B_aug[out_idx, :n_z, :] = fitcoef[:, n_z*q:n_z*q+n_u]
            
            # Bottom part: control history storage [I_{n_u}; 0; ...; 0] (only if q_u > 1)
            if q_u > 1 and n_z*q < aug_dim:
                rows_to_fill = min(aug_dim - n_z*q, b_ctrl_block.shape[0])
                B_aug[out_idx, n_z*q:n_z*q+rows_to_fill, :] = b_ctrl_block[:rows_to_fill, :]
        
        # Concatenate A_aug and B_aug
        AB_aug = np.concatenate((A_aug, B_aug), axis=2)
        
        return AB_aug