#!/usr/bin/env python

import numpy as np
# from model_free_DDP import DDP
from POM_model_free_DDP import DDP
import time
# from ltv_sys_id import ltv_sys_id_class
from arma_ltv_sys_id import ltv_sys_id_class
from material_params import *
import os
import math
import json
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
# import pde.burgers as sim
# import PFM

class model_free_material_DDP(DDP, ltv_sys_id_class):
	
	def __init__(self, initial_state, final_state, MODEL_XML, alpha, horizon, state_dimension, control_dimension, Q, Q_final, R):
		
		'''
			Declare the matrices associated with the cost function
		'''
		self.Q = Q
		self.Q_final = Q_final
		self.R = R


		DDP.__init__(self, None, state_dimension, control_dimension, alpha, horizon, initial_state, final_state)
		ltv_sys_id_class.__init__(self, None, state_dimension, control_dimension, n_substeps, n_samples=feedback_n_samples)

	def state_output(self, state):
		'''
			Given a state in terms of Mujoco's MjSimState format, extract the state as a numpy array, after some preprocessing
		'''
		
		return state#np.concatenate([state.qpos, state.qvel]).reshape(-1, 1)


	def cost(self, x, u):
		'''
			Incremental cost in terms of state and controls
		'''

		z = np.zeros((np.shape(self.Q)[0], 1))
		z[:obs_dimension,:] = C @(x - self.X_g)
		return ((z.T @ self.Q) @ z) + (((u.T) @ self.R) @ u)
		# return (9*(np.mean(x)-np.mean(self.X_g))**2 + (((u.T) @ self.R) @ u))
	
	def cost_final(self, x):
		'''
			Cost in terms of state at the terminal time-step
		'''
		z = np.zeros((np.shape(self.Q)[0], 1))
		z[:obs_dimension,:] = C @(x - self.X_g)
		return ((z.T @ self.Q_final) @ z)
		# return 9000*(np.mean(x)-np.mean(self.X_g))**2

	def initialize_traj(self, path=None, init=None):#Add check for replan, and add an input for U_0 as initial guess
		'''
		Initial guess for the nominal trajectory by default is produced by zero controls
		'''

		
		if path is None:
			
			for t in range(0, self.N):
				#print(np.shape(np.random.normal(0, nominal_init_stddev, (self.n_u, 1))))
				if init is None:
					self.U_p[t, :] = np.random.normal(0, nominal_init_stddev, (self.n_u, 1))	#np.random.normal(0, 0.01, self.n_u).reshape(self.n_u,1)#DM(array[t, 4:6])
					if t<max(q,q_u):
						self.U_p[t,:] = 0*self.U_p[t,:]
				else:
					self.U_p[t, :] = init[t,:]
				
			np.copyto(self.U_p_temp, self.U_p)
			
			self.forward_pass_sim()
			
			np.copyto(self.X_p_temp, self.X_p)

		else:

			array = np.loadtxt('../rrt_path.txt')
			### INCOMPLETE

	def replan(self, init_state, t1, horizon, ver, cost):

		path_to_model_free_DDP = "/home/karthikeya/D2C-2.0"
		MODEL_XML = path_to_model_free_DDP + "/models/fish_old.xml"
		path_to_exp = path_to_model_free_DDP + "/experiments/Material/feedback_tests/exp_replan"

		path_to_file = path_to_exp + "/material_policy_"+str(ver-1)+".txt"
		path_to_file_re = path_to_exp + "/material_policy_"+str(ver)+".txt"
		training_cost_data_file = path_to_exp + "/training_cost_data.txt"
		path_to_data = path_to_exp + "/material_D2C_DDP_data.txt"

		print(path_to_file)
		
		n_iterations=5
		alpha = .7
		U_p_rep = np.zeros((horizon, self.n_u, 1))
		xdim = int(math.sqrt(self.n_x))
		check_flag=True
		if horizon==2:
			check_flag=False

		with open(path_to_file) as f:

				Pi = json.load(f)

		for i in range(0, horizon):
			U_p_rep[i, :] = (np.array(Pi['U'][str(i)])).flatten().reshape(np.shape(U_p_rep[i, :]))


		#if ver==1:
		#	print('Umax = '+str(np.max(U_p_rep)))

		#print('Init = '+str(U_p_rep[0,:]))
		# print('Ver = ',ver)
		#print('Init = ',init_state)
		#print('Final = ',self.X_g)
		# print('New horizon =',horizon-1)
		#print('State_dim = ',self.n_x)
		#print('Control_dim = ',self.n_u)
		
		updated_model = model_free_material_DDP(init_state, self.X_g, MODEL_XML, alpha, horizon-1, self.n_x, self.n_u, Q, Q_final, R)
		# print('Check = ', horizon-1)
		updated_model.iterate_ddp(n_iterations, finite_difference_gradients_flag=False, u_init=None)#U_p_rep[t1:, :])
		# print('Check2 = ', horizon-1)
		#print('ctrl = ', updated_model.U_p[0])
		#updated_model.plot_episodic_cost_history(save_to_path=path_to_exp+"/replan_episodic_cost_training_step"+str(ver)+".png")
		updated_model.save_policy(path_to_file_re)
		# print('CheckS = ', horizon-1)

		testCL = updated_model.test_episode(1, path=path_to_file_re, noise_stddev=noise_std_test, check_plan = check_flag, init=init_state, version=ver, cst=cost)
		if check_flag==False:
			print("Ver + "+str(ver)+" Replanned Goal diff = "+str(np.linalg.norm(testCL-final_state)))
			print("Final state = ", testCL.reshape((xdim,xdim)))
			#print('Total rep cost = '+str(cost))


if __name__=="__main__":

	# Path of the model file
	path_to_model_free_DDP = "/home/karthikeya/POD2C"
	MODEL_XML = path_to_model_free_DDP + "/models/fish_old.xml"
	path_to_exp = path_to_model_free_DDP + "/experiments/Material/exp_atm"

	path_to_file = path_to_exp + "/material_policy_0.txt"
	training_cost_data_file = path_to_exp + "/training_cost_data.txt"
	path_to_data = path_to_exp + "/material_D2C_DDP_data.txt"


	# Declare other parameters associated with the problem statement
	
	# alpha is the line search parameter during forward pass
	alpha = .7

	# Declare the initial state and the final state in the problem

	sdim = int(math.sqrt(state_dimension))
	cdim = int(math.sqrt(control_dimension/2))
	initial_state = np.zeros((sdim, sdim))
	
	# final_state = np.ones((sdim,sdim))#np.loadtxt('Final.txt')#0.9999*np.ones([sdim, sdim])#.reshape((state_dimension*state_dimension, ))
	# final_state[int(sdim/2):,0:int(sdim/2)]=-1
	# final_state[0:int(sdim/2),int(sdim/2):]=-1
	# goal_conc = 0.3
	# final_state = goal_conc*np.ones((sdim,sdim))
	# final_state = np.load('relaxed_init_ch.npy')

	# #BANDED STATE 10X10
	# for x in range(int(sdim/10)):
	# 	final_state[:, 10*x+3] = -1
	# 	final_state[:, 10*x+2] = -1
	# 	final_state[:, 10*x+6] = -1
	# 	final_state[:, 10*x+7] = -1


	# initial_state = np.mean(final_state)*np.ones((sdim, sdim))
	
	#CHECKERBOARD STATE 20X20
	# for x in range(int(sdim/2)):
	# 	for y in range(int(sdim/2)):
	# 		final_state[2*x,2*y]=(-1)**(x+y)
	# 		final_state[2*x+1,2*y]=(-1)**(x+y)
	# 		final_state[2*x,2*y+1]=(-1)**(x+y)
	# 		final_state[2*x+1,2*y+1]=(-1)**(x+y)

	# for x in range(int(sdim/5)):
	# 	for y in range(int(sdim/5)):
	# 		final_state[5*x:5*x+5,5*y:5*y+5]=(-1)**(x+y)
			
	# ATM STATE 50X50		
	final_state=np.loadtxt('Final.txt')

	
	#final_state=final_state.astype(np.float)
	#final_state = -0.9999*np.ones(1)

	# initial_state = np.mean(final_state)*np.ones((sdim, sdim))
	# initial_state = initial_state.reshape((state_dimension, 1))

	# plt.matshow(final_state)
	# plt.show()
	

	print('Initial phase : \n', initial_state)
	print('Goal phase : \n', final_state)
	final_state = final_state.reshape((state_dimension, 1))
	# No. of ILQR iterations to run
	n_iterations =15#5#40

	# Initiate the above class that contains objects specific to this problem
	model = model_free_material_DDP(initial_state, final_state, MODEL_XML, alpha, horizon, state_dimension, control_dimension, Q, Q_final, R)


	# ---------------------------------------------------------------------------------------------------------
	# -----------------------------------------------Training---------------------------------------------------
	# ---------------------------------------------------------------------------------------------------------
	# Train the policy

	training_flag_on = True

	if training_flag_on:

		with open(path_to_data, 'w') as f:

			f.write("D2C training performed for a micro-structure control task:\n\n")

			f.write("System details : {}\n".format(os.uname().sysname + "--" + os.uname().nodename + "--" + os.uname().release + "--" + os.uname().version + "--" + os.uname().machine))
			f.write("-------------------------------------------------------------\n")

		time_1 = time.time()

		# Run the DDP algorithm
		# To run using our LLS-CD jacobian estimation (faster), make 'finite_difference_gradients_flag = False'
		# To run using forward difference for jacobian estimation (slower), make 'finite_difference_gradients_flag = True'

		model.iterate_ddp(n_iterations, finite_difference_gradients_flag=False)
		
		time_2 = time.time()

		D2C_algorithm_run_time = time_2 - time_1

		print("D2C-2 algorithm run time taken: ", D2C_algorithm_run_time)

		# Save the history of episodic costs 
		with open(training_cost_data_file, 'w') as f:
			for cost in model.episodic_cost_history:
				f.write("%s\n" % cost)

		# Test the obtained policy
		print('pmtest = ',model.p_mask)
		model.save_policy(path_to_file)
		#U_nominal = U_p
		#X_nominal = X_p
		#np.save('X_nominal.npy', X_nominal)

		with open(path_to_data, 'a') as f:																																																																																									

				f.write("\nTotal time taken: {}\n".format(D2C_algorithm_run_time))
				f.write("------------------------------------------------------------------------------------------------------------------------------------\n")

		# Display the final state in the deterministic policy on a noiseless system
		print(model.X_p[-1].reshape(sdim,sdim))

		ax = plt.subplot()
		im = ax.matshow(model.X_p[-1].reshape((sdim, sdim)), cmap='RdBu')
		plt.colorbar(im)

		# plt.matshow(model.X_p[-1].reshape(sdim, sdim))
		# plt.show()
		print('MSE = ', np.linalg.norm(model.X_p[-1].reshape(sdim,sdim)-final_state.reshape(sdim,sdim)),2)
		# print(np.shape(model.X_p))
		# np.save('relaxed_cb.npy',model.X_p)
		# np.save('Unrelaxed_fin_ch.npy',model.X_p[-1].reshape([sdim, sdim]))
		
		# Plot the episodic cost during the training
		model.plot_episodic_cost_history(save_to_path=path_to_exp+"/episodic_cost_training.png")
		plt.figure()
		plt.plot(np.linspace(-1, 1, 10), initial_state, 'b')
		plt.plot(np.linspace(-1, 1, 10), model.X_p[5,:], '--r')
		plt.plot(np.linspace(-1, 1, 10), model.X_p[-1], 'g')
		plt.legend(['t_0', 't_i', 't_f'])
		plt.xlabel('X')
		plt.ylabel('Velocity')
		# plt.show()

		plt.figure()
		plt.plot(np.arange(horizon), model.U_p[:,0], 'b')
		plt.plot(np.arange(horizon), model.U_p[:,1], 'r')
		# plt.plot(np.linspace(-1, 1, 10), model.X_p[-1], 'g')
		plt.legend(['u_in', 'u_out'])
		plt.xlabel('t')
		plt.ylabel('Control')
		plt.show()


		state_history_nominal=model.X_p[-1]
		#for t_time in range(1):#range(np.shape(state_history_nominal)[0]):
		#phi=state_history_nominal[t_time,:].reshape([20, 20])
	t_time=0
	phi=initial_state.reshape([sdim, sdim])#final_state.reshape([sdim, sdim])#state_history_nominal.reshape([sdim, sdim])
	ims = []
	x=(np.arange(int(math.sqrt(state_dimension))))*0.1
	y=(np.arange(int(math.sqrt(state_dimension))))*0.1
	X,Y = np.meshgrid(x,y)
	fig = plt.figure(figsize=(10, 6.5))
	ax = fig.add_subplot(121.5)
	CS = ax.contourf(X, Y, phi, cmap=plt.cm.get_cmap('RdBu'))# v, cmap=plt.cm.get_cmap('RdBu'))
	ax.set_title('phase field evolution')
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_aspect(1.0)
	axs=[ax]
	fig.suptitle('Explicit | 50x50', position=(0.2, 0.94), fontsize=20, fontweight='bold')
	#if t_time <= 5:
	#	fig.text(0.15, 0.84, 'T = -2, h = 0', fontsize=16)
	#else:
	#	fig.text(0.02, 0.84, 'T = rand(-2,2), h = rand(-1,0)', fontsize=16)
		
	fig.text(0.35, 0.915, 'Step = %f' % t_time, fontsize=16, bbox={'facecolor': 'yellow', 'alpha': 0.3})
			
	ims.append([CS.collections,])

	idx=int(t_time)
	savefile=path_to_exp+'/file_final_' + str(idx) + '.png'
	plt.savefig(savefile, dpi=300)
	plt.draw()
	plt.pause(0.002)
	plt.close()


	#print(model.test_episode(render=0, path=path_to_file, noise_stddev=0).reshape((sdim,sdim)))
	#print(np.sum(model.test_episode(render=0, path=path_to_file, noise_stddev=0)))
	'''mseval=np.zeros(np.shape(model.X_p)[0])
	print(model.U_p[5])
	for t in range(0,horizon-1):
		#if t is 0:
		mseval[t] = model.cost(model.X_p[t], model.U_p[t+1])#(((model.U_p[t].T) @ R) @ model.U_p[t])
		#else:
		#	mseval[t] = mseval[t-1]+model.cost(model.X_p[t], model.U_p[t+1])#(((model.U_p[t].T) @ R) @ model.U_p[t])
	#mseval[t]+= model.cost_final(model.X_p[t])
		

	print(0.025*np.arange(10),mseval)
	plt.plot(mseval)
	plt.show()'''
	'''with open(path_to_file) as f:
		Pi = json.load(f)

	xdim=10
	U_p = np.zeros([horizon, 2*xdim*xdim])
	#X_p_t = np.zeros([horizon, xdim*xdim])
	#X_p_t[-1] = np.zeros((xdim*xdim))
	#ctrl=np.zeros((2*xdim*xdim, ))
	#K_t = np.zeros(np.shape(model.K))
	#X_t = np.zeros([horizon, xdim*xdim])
	#print(np.shape(K_t))

	for t in range(0, horizon-1):
		U_p[t]=np.array(Pi['U'][str(t)]).flatten()
	#	K_t[t]=np.array(Pi['K'][str(t)])
	#	X_t[t]=np.array(Pi['X'][str(t)]).flatten()

	U_p = np.repeat(U_p, int(100/horizon), axis=0)
	#K_t = np.repeat(K_t, int(100/horizon), axis=0)
	#X_t = np.repeat(X_t, int(100/horizon), axis=0)
	control_cost=0
	for t in range(100):
		control_cost+= (U_p[t].T @ R) @ U_p[t]
	print(control_cost)'''
	#print(U_p[:,0])
		
	#for t in range(horizon-1):
		#if t==0:
		#ctrl[:] = (U_p[t+1]).flatten()
		#else:
		#	ctrl[:] = (U_p[t]+K_t[t-1] @ (X_p_t[t-1] - X_t[t-1])).flatten()

	

		#X_p_t[t] = PFM.simulation(500, 0.001, 0.001, ctrl[:int(len(ctrl)/2)].reshape((xdim, xdim)), ctrl[int(len(ctrl)/2):].reshape((xdim, xdim)), X_p_t[t-1].reshape((xdim, xdim))).reshape((xdim*xdim))
		#print(X_p_t[t].reshape((10,10)))

	#print(np.array(Pi['X'][str(4)]).reshape((10,10)))
	#print(model.test_episode(render=0, path=path_to_file).reshape(10,10))
	#print(X_p_t[1].reshape((10,10)))
	#print(X_p_t[horizon-2].reshape((10,10)))'''


	# ---------------------------------------------------------------------------------------------------------
	# -----------------------------------------------Testing---------------------------------------------------
	# ---------------------------------------------------------------------------------------------------------
	# Test the obtained policy

	test_flag_on = False
	#np.random.seed(1)

	if test_flag_on:

		f = open(path_to_exp + "/material_testing_data.txt", "a")

		def frange(start, stop, step):
			i = start
			while i < stop:
				yield i
				i += step
		
		u_max = 3.5

		try:

			for i in frange(0.0, 1.02, 0.02):

				print(i)
				print("\n")
				terminal_mse = 0
				Var_terminal_mse = 0
				n_samples = 100

				for j in range(n_samples):	

					terminal_state = model.test_episode(render=0, path=path_to_file, noise_stddev=i*u_max)
					terminal_mse += np.linalg.norm(terminal_state[0:3] - final_state[0:3], axis=0)
					Var_terminal_mse += (np.linalg.norm(terminal_state[0:3] - final_state[0:3], axis=0))**2

				terminal_mse_avg = terminal_mse/n_samples
				Var_terminal_mse_avg = (1/(n_samples*(n_samples-1)))*(n_samples*Var_terminal_mse - terminal_mse**2)

				std_dev_mse = np.sqrt(Var_terminal_mse_avg)

				f.write(str(i)+",\t"+str(terminal_mse_avg[0])+",\t"+str(std_dev_mse[0])+"\n")
		except:

			print("Testing failed!")
			f.close()

		f.close()

	xdim=int(math.sqrt(state_dimension))

	# feedback_costs = np.array([])
	# # replan_costs = np.array([])
	# np.save('std8_noise_nfb.npy',feedback_costs)
	# np.save('replan_costs_std8_noise.npy',replan_costs)

	# end_disc = np.array([])
	# np.save('end_disc.npy',end_disc)
	# for i in range(1):
	# 	testCL = model.test_episode(1, path=path_to_file, noise_stddev=noise_std_test, check_plan=False, init=initial_state)
	# 	print("Goal diff = "+str(np.linalg.norm(testCL-final_state)))
	# AA=np.load('end_disc.npy')
	# np.save('end_disc.npy',np.append(AA, np.linalg.norm(testCL-final_state)))
	# # plt.matshow(testCL.reshape((sdim, sdim)))
	#plt.show()
	# for i in range(1):
	# 	testCL2 = model.test_episode(1, path=path_to_file, noise_stddev=noise_std_test, check_plan=True, init=initial_state)
	
	#print("Goal diff replanned = "+str(np.linalg.norm(testCL2-final_state)))
	'''with open(path_to_file) as f:
		Pi = json.load(f)

	Umax=-99999
	for i in range(10):
		Umax = max(Umax, max(np.array(Pi['U'][str(i)]))) 

	print(Umax)

	Na = 11
	Nb = 10
	B = np.zeros(Nb)
	A = np.zeros(Na)
	Astd=np.zeros(Na)

	B_cl = np.zeros(Nb)
	A_cl = np.zeros(Na)
	Astd_cl=np.zeros(Na)
	
	xdim=10
	U_p = np.zeros([10, 2*xdim*xdim])
	X_p = model.X_p
	X_p[-1] = -0.9999*np.zeros((xdim*xdim, 1))
	
	ctrl=np.zeros((2*xdim*xdim, ))

	eps = []
	for i in range(Na):
		for j in range(Nb):
			#testCL = model.test_episode(1, path=path_to_file, noise_stddev=0.1*i)

			''''''X_p[-1] = -0.9999*np.zeros((state_dimension, 1))+np.random.normal(np.zeros(np.shape(X_p[-1])), 0.1*i)
			for t in range(0, 10):
				U_p[t]=np.array(Pi['U'][str(t)]).flatten()

				ctrl[:] = U_p[t].flatten()

				X_p[t] = PFM.simulation(25, 0.001, 0.001, ctrl[:int(len(ctrl)/2)].reshape((xdim, xdim)), ctrl[int(len(ctrl)/2):].reshape((xdim, xdim)), X_p[t-1].reshape((xdim, xdim))).reshape((xdim*xdim, 1))
			#test = model.test_episode(1, path=None, noise_stddev=0.1*i)#path_to_file, noise_stddev=0.1*i*Umax)
			#test = swimmer6.state_output(swimmer6.test_episode(1, err=0.1*i, ch=True))''''''
			
			
			#B[j] = np.linalg.norm(testCL-final_state)#math.sqrt((test-final_state).T.dot(test-final_state))#np.linalg.norm(test-final_state)#((test-final_state).T.dot(test-final_state))/len(test)#
			#B_cl[j] = np.linalg.norm(testCL-final_state)
		#A[i] = (np.mean(B))
		#Astd[i] = (np.std(B))

		#A_cl[i] = (np.mean(B_cl))
		#Astd_cl[i] = (np.std(B_cl))
		#print(np.max(test))
		#print(((test-final_state).T.dot(test-final_state))/len(test))
		
		print(i)
		
		eps.append(10*i)

	#print(A)
				
	#np.save('MeanCL.npy', A_cl)
	#np.save('StdCL.npy', Astd_cl)
	#np.save('MeanOL.npy', A)
	#np.save('StdOL.npy', Astd)
	A_cl=np.load('MeanCL.npy')
	Astd_cl=np.load('StdCL.npy')
	A=np.load('MeanOL.npy')
	Astd=np.load('StdOL.npy')

	plt.plot(eps, A, color='r')
	plt.fill_between(eps, A-Astd, A+Astd, alpha=0.3, color='r')

	plt.plot(eps, A_cl, color='b')
	plt.fill_between(eps, A_cl-Astd_cl, A_cl+Astd_cl, alpha=0.3)
	#print(A)
	plt.xlabel('Std of perturbed noise (Percent of max phi init i.e. 1)')
	plt.ylabel('L2 norm of terminal state error')
	plt.legend(['No feedback','With feedback'])
	plt.show()'''

	'''np.save('MatmeanOL.npy', A)
	np.save('MatstdOL.npy', Astd)
	meanCL =np.load('Matmean.npy')
	stdCL = np.load('Matstd.npy')
	plt.plot(eps, A, color='g')
	plt.fill_between(eps, A-Astd, A+Astd, alpha=0.3)

	plt.plot(eps, meanCL, color='b')
	plt.fill_between(eps, A_cl-Astd_cl, A_cl+Astd_cl, alpha=0.3)
	#print(A)
	plt.xlabel('Std of perturbed noise (Percent of max control)')
	plt.ylabel('L2 norm of terminal state error')
	plt.show()
	#print(model.test_episode(1, path=path_to_file, noise_stddev=0.5).reshape([10, 10]))
	#model.test_episode(1, noise_stddev=0.1*0*Umax, printch=True)
	with open(path_to_file) as f:
		Pi = json.load(f)

	mse=np.zeros(100)
	state_history_nominal = model.test_episode(1, path=path_to_file, noise_stddev=0)
	for i in range(100):
		mse[i] = np.linalg.norm(state_history_nominal[i, :]-final_state, 2)
	plt.plot(np.arange(100), mse)
	plt.show()
	print(np.shape(state_history_nominal))

	for t_time in range(1):#range(np.shape(state_history_nominal)[0]):
		#phi=state_history_nominal[t_time,:].reshape([20, 20])
		phi=state_history_nominal.reshape([20, 20])
		ims = []
		x=(np.arange(int(math.sqrt(state_dimension))))*0.1
		y=(np.arange(int(math.sqrt(state_dimension))))*0.1
		X,Y = np.meshgrid(x,y)
		fig = plt.figure(figsize=(10, 6.5))
		ax = fig.add_subplot(121.5)
		CS = ax.contourf(X, Y, phi, cmap=plt.cm.get_cmap('RdBu'))# v, cmap=plt.cm.get_cmap('RdBu'))
		ax.set_title('phase field evolution')
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_aspect(1.0)
		axs=[ax]
		fig.suptitle('Explicit | 20x20', position=(0.2, 0.94), fontsize=20, fontweight='bold')
		#if t_time <= 5:
		#	fig.text(0.15, 0.84, 'T = -2, h = 0', fontsize=16)
		#else:
		#	fig.text(0.02, 0.84, 'T = rand(-2,2), h = rand(-1,0)', fontsize=16)
			
		fig.text(0.35, 0.915, 'Step = %f' % t_time, fontsize=16, bbox={'facecolor': 'yellow', 'alpha': 0.3})
				
		ims.append([CS.collections,])

		idx=int(t_time)
		savefile=path_to_exp+'/Images/file' + str(idx) + '.png'
		plt.savefig(savefile, dpi=300)
		plt.draw()
		plt.pause(0.002)
		plt.close()'''

	##########################
	'''if True:
		T_m1=np.zeros(100)
		T_1=np.zeros(100)
		h_m1=np.zeros(100)
		h_1=np.zeros(100)
		U_p = np.zeros([100, 800])
		X_p = model.X_p
		print(np.shape(model.X_p))
		X_p[-1] = -0.9999*np.zeros((400, 1))
		xdim=20
		ctrl=np.zeros((800, ))
		mse2=np.zeros(100)
		mse1=np.zeros(100)

		with open("/home/karthikeya/D2C-2.0/experiments/Material/exp_11_i-1v2/material_policy.txt") as f:
			Pi = json.load(f)

		for t in range(20):
			T_m1[t]=np.array(Pi['U'][str(t)])[0]
			h_m1[t]=np.array(Pi['U'][str(t)])[1]

		with open("/home/karthikeya/D2C-2.0/experiments/Material/exp_11_i1v2/material_policy.txt") as f2:
			Pi2 = json.load(f2)

		for t in range(20):
			T_1[t]=np.array(Pi2['U'][str(t)])[0]
			h_1[t]=np.array(Pi2['U'][str(t)])[1]

		#print('Check')
		#print(self.U_p[:, final_state==0.9999])
		for t in range(0, 20):
			#print(np.shape(np.random.normal(0, nominal_init_stddev, (self.n_u, 1))))
			#self.U_p[t, :] = np.random.normal(0, nominal_init_stddev, (self.n_u, 1))	#np.random.normal(0, 0.01, self.n_u).reshape(self.n_u,1)#DM(array[t, 4:6])
			for x in range(400):
				if final_state[x]==0.9999:
					U_p[t, x]=T_1[t]
					U_p[t, 400+x]=h_1[t]
				if final_state[x]==-0.9999:
					U_p[t, x]=T_m1[t]
					U_p[t, 400+x]=h_m1[t]

			#U_p[t]=np.array(Pi['U'][str(t)]).flatten()

			ctrl[:] = U_p[t].flatten()

			X_p[t] = PFM.simulation(25, 0.001, 0.001, ctrl[:int(len(ctrl)/2)].reshape((xdim, xdim)), ctrl[int(len(ctrl)/2):].reshape((xdim, xdim)), X_p[t-1].reshape((xdim, xdim))).reshape((xdim*xdim, 1))
			mse2[t]=np.linalg.norm(X_p[t]-final_state)
			mse1[t]=np.linalg.norm(X_p[t]-final_state)

		for t in range(20, 100):
			U_p[t,0:400]=-2*np.ones(400)
			U_p[t,400:]=np.zeros(400)

			ctrl[:] = U_p[t].flatten()

			X_p[t] = PFM.simulation(25, 0.001, 0.001, ctrl[:int(len(ctrl)/2)].reshape((xdim, xdim)), ctrl[int(len(ctrl)/2):].reshape((xdim, xdim)), X_p[t-1].reshape((xdim, xdim))).reshape((xdim*xdim, 1))
			mse2[t]=np.linalg.norm(X_p[t]-final_state)

		plt.matshow(X_p[100-1].reshape((20,20)))
		plt.matshow(X_p[20-1].reshape(20,20))

		for t in range(20, 100):
			U_p[t,0:400]=-2*np.ones(400)
			U_p[t,400:]=np.zeros(400)

			ctrl[:] = U_p[t].flatten()

			X_p[t] = PFM.simulation(25, 0.0001, 0.001, ctrl[:int(len(ctrl)/2)].reshape((xdim, xdim)), ctrl[int(len(ctrl)/2):].reshape((xdim, xdim)), X_p[t-1].reshape((xdim, xdim))).reshape((xdim*xdim, 1))
			mse1[t]=np.linalg.norm(X_p[t]-final_state)

		#plt.figure()
		plt.matshow(X_p[100-1].reshape((20,20)))
		#np.save('mse_singlev3.npy',mse2)
		plt.figure()
		plt.plot(np.arange(100),mse2)
		plt.plot(np.arange(100),mse1)
		plt.legend(['constant gamma', 'gamma2=0.1*gamma1'])
		plt.show()
		mse1=np.load('mse_singlev2.npy')
		mse2=np.load('mse_iteratedv2.npy')
		plt.plot(np.arange(20), mse2[:20])
		plt.plot(np.arange(20), mse1[:20])
		plt.legend(['Traj with single pt u', 'Traj with init guess= single-pt'])
		plt.xlabel('Timesteps')
		plt.ylabel('MSE wrt Final State')
		plt.show()'''