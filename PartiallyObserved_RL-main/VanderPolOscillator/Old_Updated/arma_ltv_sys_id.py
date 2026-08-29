#!/usr/bin/env python

import numpy as np
import scipy.linalg.blas as blas
#from PFM import *
from material_params import *
#from mujoco_py import load_model_from_path, MjSim, MjViewer
import math
# import burgers_test as simulator
import pde.burgers as simulator
# import pde.burgers_corr as simulator
import sys
import pde.PFM as PFM
import pde.PFM_CH as PFM_CH



class ltv_sys_id_class(object):

	def __init__(self, model_xml_string,  state_size, action_size, n_substeps=1, n_samples=500):

		self.n_x = state_size
		self.n_u = action_size

		# Standard deviation of the perturbation 
		self.sigma = 1e-7
		self.n_samples = n_samples

		#GSo Mask for actuation
		self.xdim = int(math.sqrt(self.n_x))
		

		#self.sim = ACsolver()#MjSim(load_model_from_path(model_xml_string), nsubsteps=n_substeps)
		


	def sys_id(self, x_0, u_nom, central_diff, activate_second_order=0, V_z=None, p_mask= None, m_mask= None):

		'''
			system identification for a given nominal state and control
			returns - a numpy array with F_x and F_u horizantally stacked
		'''
		################## defining local functions & variables for faster access ################
		generate_nominal_traj = self.generate_nominal_traj
		simulate = self.forward_simulate
		n_x = self.n_x
		n_u = self.n_u
		n_z = np.shape(C)[0]

		if p_mask is not None:
			self.p_mask = p_mask
			self.m_mask = m_mask
		# print('Shape of X_t = ', np.shape(x_t))

		##########################################################################################
		
		A_aug = np.zeros((self.n_samples, n_z*q+n_u*(q_u-1), n_z*q+n_u*(q_u-1)))
		B_aug = np.zeros((self.n_samples, n_z*q+n_u*(q_u-1), n_u))
		

		# Generating nominal traj
		Z_norm = C @ generate_nominal_traj(x_0, u_nom)
		

		# u_max = np.max(abs(u_nom))
		# U_ = 0.2*u_max*np.random.normal(0, self.sigma, (self.n_samples, n_u*(horizon+1)))
		
		# delta_z = np.zeros((self.n_samples, n_z*(horizon+2)))
		# X = np.zeros((n_x, horizon+1))
		
		# ctrl = np.zeros((n_u, 1))

		# # Generating delta_z for all rollouts
		# for j in range(self.n_samples):
		# 	X[:,0] = x_0.reshape((n_x,))
		# 	for i in range(horizon):
		# 		ctrl[:] = u_nom[i] + U_ [j, n_u*(horizon-i)+1:n_u*(horizon-i+1)]
		# 		X[:, i+1] = simulate(None, X[:,i], ctrl).reshape(n_x,)
		# 		delta_z[j, n_z*(horizon-i)+1:n_z*(horizon-i+1)+1] = (C @ X[:,i+1]) - Z_norm[:,i+1]


		u_max = np.max(abs(u_nom))
		U_ = 0.1*u_max*np.random.normal(0, self.sigma, (self.n_samples, n_u*(horizon+1)))
		
		delta_z = np.zeros((self.n_samples, n_z*(horizon+1)))
		X = np.zeros((n_x, horizon+1))
		
		ctrl = np.zeros((n_u, 1))

		# Generating delta_z for all rollouts
		for j in range(self.n_samples):
			X[:,0] = x_0.reshape((n_x,))
			for i in range(horizon):
				ctrl[:] = u_nom[i] + U_ [j, n_u*(horizon-i):n_u*(horizon-i+1)].reshape(np.shape(u_nom[i]))
				X[:, i+1] = simulate(None, X[:,i], ctrl).reshape(n_x,)

				testsum1 = np.sum(X[:, i+1])
				if np.isnan(testsum1):
					print('X has nan')
					print('X0 = ', x_0)
					# print('Nominal Control : ',u_nom)
					# print('Control = ', ctrl)
					print('X [',i,'] = ', X[:, i+1])
					sys.exit()
				# delta_z[j, n_z*(i+1):n_z*(i+2)] = (C @ X[:,i+1]) - Z_norm[:,i+1]
				delta_z[j, n_z*(horizon-i-1):n_z*(horizon-i)] = (C @ X[:,i+1]) - Z_norm[:,i+1]



		fitcoef=np.zeros((n_z,n_z*q+n_u*q_u,horizon)); # M1 * fitcoef = delta_z

		
		for i in range(max(q, q_u),horizon):
			
			# M1 = np.hstack([delta_z[:, n_z*(horizon-i+1)+1:n_z*(horizon-i+q+1)], U_[:, n_u*(horizon-i+1)+1:n_u*(horizon-i+q_u+1)]])
			M1 = np.hstack([delta_z[:, n_z*(horizon-i):n_z*(horizon-i+q)], U_[:,n_u*(horizon-i):n_u*(horizon-i+q_u)]])
			delta = delta_z[:, n_z*(horizon-i-1):n_z*(horizon-i)]

		
			
			
			mat, res, rank, S = np.linalg.lstsq(M1, delta, rcond=None)
			# print(np.shape(mat))
			fitcoef[:, :, i] =  mat.T
			A_aug[i, :np.shape(fitcoef)[0],:] =  np.hstack((fitcoef[:, :n_z*q,i], fitcoef[:, n_z*q+n_u:, i]))
			A_aug[i, np.shape(fitcoef)[0]:np.shape(fitcoef)[0]+n_z*(q-1),:] = np.hstack((np.eye(n_z*(q-1)), np.zeros((n_z*(q-1), n_z+n_u*(q_u-1)))))
			# print(np.shape(A_aug[i, np.shape(fitcoef)[0]+n_z*(q-1):,:]))
			A_aug[i, np.shape(fitcoef)[0]+n_z*(q-1):,:] = np.zeros(np.shape(A_aug[i, np.shape(fitcoef)[0]+n_z*(q-1):,:]))
			A_aug[i, np.shape(fitcoef)[0]+n_z*(q-1)+n_u:,n_z*q+1:n_z*q+1+n_u*(q_u-2)] = np.eye(n_u*(q_u-2)) 

			# print(np.shape(fitcoef[:,n_z*q:n_z*q+n_u,i]))
			B_aug[i, :np.shape(fitcoef)[0]+n_z*(q-1), :] = np.vstack([fitcoef[:,n_z*q:n_z*q+n_u,i], np.zeros((n_z*(q-1), n_u))])
			B_aug[i, np.shape(fitcoef)[0]+n_z*(q-1):, :] = np.vstack([np.eye(n_u*min(1, q_u-1)), np.zeros((n_u*(q_u-2), n_u))])
			# sys.exit()

		V_x_F_XU_XU = None


		###############################################################################
		# print(delta_z[:, n_z*(i-q):n_z*(i)].reverse())
		# sys.exit()

		# TEST_NUM=1; #number of monte-carlo runs to verify the fitting result
		# ucheck=0.01*u_max*np.random.random((n_u*(horizon+1), TEST_NUM))#randn(IN_NUM*(horizon+1),TEST_NUM); % input used for checking
		# y_sim=np.zeros(n_z*(horizon+1),TEST_NUM); # output from real system
		# y_pred=zeros(n_z*(horizon+1),TEST_NUM); # output from arma model
		# y_sim = C @ generate_nominal_traj(x_0, u_nom+ucheck)
		
		# for j in range(TEST_NUM):
		# 	X[:,0] = x_0.reshape((n_x,))
		# 	for i in range(horizon):
		# 		ctrl[:] = u_nom[i] + ucheck[n_u*i:n_u*(i+1), :].reshape(np.shape(u_nom[i]))
		# 		X[:, i+1] = simulate(None, X[:,i], ctrl).reshape(n_x,)
		# 		y_sim[n_z*(i+1):n_z*(i+2), j] = (C @ X[:,i+1])# - Z_norm[:,i+1]

		# y_pred[:,:]=y_sim(n_u*(horizon-q-1)+1:n_z*(horizon+1),:); # manually match the first few steps
		# for i=max(q,qu)+2:1:horizon % start to apply input after having enough data
		#     M2=[y_pred(n_z*(horizon-i+1)+1:n_z*(horizon-i+1+q),:);ucheck(IN_NUM*(horizon-i+1)+1:IN_NUM*(horizon-i+qu+1),:)];
		#     y_pred(n_z*(horizon-i)+1:n_z*(horizon-i+1),:)=fitcoef(:,:,i)*M2;
		# end
		###############################################################################

		
		# return F_ZU, V_x_F_XU_XU
		return A_aug, B_aug, V_x_F_XU_XU, X




	def sys_id_FD(self, x_t, u_t, central_diff, activate_second_order=0, V_x_=None):

		'''
			system identification by a forward finite-difference for a given nominal state and control
			returns - a numpy array with F_x and F_u horizantally stacked
		'''
		################## defining local functions & variables for faster access ################

		simulate = self.simulate
		n_x = self.n_x
		n_u = self.n_u
		##########################################################################################
		
		z_t = C @ x_t
		X2U = np.random.normal(0.0, self.sigma, (n_x, n_x + n_u))
		X_ = np.copy(XU[:, :n_x])
		U_ = np.copy(XU[:, n_x:])

		F_XU = np.zeros((n_x, n_x + n_u))
		V_x_F_XU_XU = None

		x_t_next = simulate(x_t.T, u_t.T)

		if central_diff:
			
			for i in range(0, n_x):
				for j in range(0, n_x):

					delta = np.zeros((1, n_x))
					delta[:, j] = XU[i, j]
					
					F_XU[i, j] = (simulate(x_t.T + delta, u_t.T)[:, i] - x_t_next[:, i])/XU[i, j]

			for i in range(0, n_x):
				for j in range(0, n_u):

					delta = np.zeros((1, n_u))
					delta[:, j] = XU[i, n_x + j]
					F_XU[i, n_x + j] = (simulate(x_t.T , u_t.T + delta)[:, i] - x_t_next[:, i])/XU[i, n_x + j]
					
		else:

			Y = (simulate((x_t.T) + X_, (u_t.T) + U_) - simulate((x_t.T), (u_t.T)))

			
	

		return F_XU, V_x_F_XU_XU	





	def simulate(self, X, U):
		
		'''
		Function to simulate a batch of inputs given a batch of control inputs and current states
		X - vector of states vertically stacked
		U - vector of controls vertically stacked
		'''
		################## defining local functions & variables for faster access ################

		#sim = self.sim
		forward_simulate = self.forward_simulate
		state_output = self.state_output

		##########################################################################################
		
		X_next = []

		# Augmenting X by adding a zero column corresponding to time
		X = np.hstack((np.zeros((X.shape[0], 1)), X))

		for i in range(X.shape[0]):

			X_next.append(state_output(forward_simulate(None, X[i] , U[i])))
			
		return np.asarray(X_next)[:,:,0]
	



	def vec2symm(self, ):
		pass




	def forward_simulate(self, sim, x, u):

		'''
			Function to simulate a single input and a single current state
			Note : The initial time is set to be zero. So, this can only be used for independent simulations
			x - append time (which is zero here due to above assumption) before state
		'''
		sdim = int(math.sqrt(state_dimension))
		#sim.set_state_from_flattened(x)
		#sim.forward()
		#sim.data.
		# ctrl = np.zeros((self.n_u, 1))
		ctrl = u
		N=sdim#self.n_x#sdim
		# dx = (2*np.pi)/self.n_x
		# dt = 1e-4
		# tperiod = 1000
		# nu = 0.1#0.01/np.pi

		# print('p_mask = ', self.p_mask)
		# sys.exit()

		## GSO ONLY##########
		ctrl_T = self.p_mask*ctrl[0]+self.m_mask*ctrl[1]#+np.random.normal(np.zeros(np.shape(self.p_mask)), process_noise_std)
		ctrl_h = self.p_mask*ctrl[2]+self.m_mask*ctrl[3]
		## GSO ONLY##########

		
		#self.sim.step(x[1:].reshape((sdim, sdim)), np.transpose(ctrl[0:int(len(ctrl)/2)]), np.transpose(ctrl[int(len(ctrl)/2):]), 25)
		#sim.render(mode='window')

		# print(simulator.simulation(tperiod, nu, dt, ctrl[:int(len(ctrl)/2)], ctrl[int(len(ctrl)/2):], x))
		
		# return simulator.simulation(tperiod, nu, dt, ctrl[:int(len(ctrl)/2)], ctrl[int(len(ctrl)/2):], x)
		# return PFM.simulation(n_ac, 0.001, 0.001, np.transpose(ctrl[0:int(len(ctrl)/2)]).reshape((sdim, sdim)), np.transpose(ctrl[int(len(ctrl)/2):]).reshape((sdim, sdim)), x.reshape((sdim, sdim))).reshape((sdim*sdim, 1))#sim.get_state()
		# return PFM_CH.simulation(n_ch, 0.001, dt_ch, np.transpose(ctrl[0:int(len(ctrl)/2)]).reshape((sdim, sdim)), np.transpose(ctrl[int(len(ctrl)/2):]).reshape((sdim, sdim)), x.reshape((sdim, sdim))).reshape((sdim*sdim, 1))#sim.get_state()
		#return PFM.simulation(25, 0.001, 0.001, np.zeros((sdim, sdim)), np.transpose(ctrl[:]).reshape((sdim, sdim)), x[1:].reshape((sdim, sdim))).reshape((sdim*sdim, 1))
		return PFM.simulation(n_ac, 0.001, 0.001, ctrl_T, ctrl_h, x.reshape((sdim, sdim))).reshape((sdim*sdim, 1))
		# return PFM_CH.simulation(n_ch, 0.001, dt_ch, ctrl_T, ctrl_h, x.reshape((sdim, sdim))).reshape((sdim*sdim, 1))
			


	def traj_sys_id(self, x_nominal, u_nominal):	
		
		'''
			System identification for a nominal trajectory mentioned as a set of states
		'''
		
		Traj_jac = []
		
		for i in range(u_nominal.shape[0]):
			
			Traj_jac.append(self.sys_id(x_nominal[i], u_nominal[i]))

		return np.asarray(Traj_jac)
		

	
	def state_output(state):

		pass


	def khatri_rao(self, B, C):
		"""
		Calculate the Khatri-Rao product of 2D matrices. Assumes blocks to
		be the columns of both matrices.

		See
		http://gmao.gsfc.nasa.gov/events/adjoint_workshop-8/present/Friday/Dance.pdf
		for more details.

		Parameters
		----------
		B : ndarray, shape = [n, p]
		C : ndarray, shape = [m, p]


		Returns
		-------
		A : ndarray, shape = [m * n, p]

		"""
		if B.ndim != 2 or C.ndim != 2:
			raise ValueError("B and C must have 2 dimensions")
		
		n, p = B.shape
		m, pC = C.shape
		
		if p != pC:
			raise ValueError("B and C must have the same number of columns")
		
		return np.einsum('ij, kj -> ikj', B, C).reshape(m * n, p)

	def generate_nominal_traj(self, x_0, u_nom):
		forward_simulate = self.forward_simulate

		Y = np.zeros((state_dimension, horizon+1))
		Y[:, 0] = x_0.reshape((state_dimension,))
		
		for i in range(horizon):
			Y[:, i+1] = forward_simulate(None, Y[:,i], u_nom[i, :]).reshape((state_dimension,))


		return Y
