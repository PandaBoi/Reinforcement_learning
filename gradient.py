import numpy as np
import matplotlib.pyplot as plt
import random
import os

cwd = os.getcwd()
# print(cwd)

"""
Gradient multi armed bandit problem of Sutton & Barto book


"""
#========================== Epsilon-Greedy MAB ============================
class Grad:
	def __init__(self,k,e,bandits,alpha,initial):

		#initializing all parameters

		self.k = k
		self.e = e
		self.alpha = alpha
		self.bandits = bandits

			

		self.k_i = np.ones((bandits,k)) # no of occ of kth arm for bth bandit
		self.pull_reward = np.zeros(bandits) #avg reward of each step
		self.q_star = np.random.normal(0,1 + initial,(bandits,k))
		self.Q_value = np.zeros((bandits,k))
		self.H_val = np.zeros((bandits,k))
		self.pi = np.zeros((bandits,k))
		self.opt_arm = np.argmax(self.q_star,1)# the highest reward arms 
		
		self.actions_taken = []
		self.pull_reward=[]
		self.opt_arm_val =0

	def select_action_g(self):
		
		self.pi[self.b] = np.exp(self.H_val[self.b])/(np.sum(np.exp(self.H_val[self.b]))) 
		self.a = np.random.choice(range(0,k), p= self.pi[self.b]) 
		self.actions_taken.append(self.a)

		if self.a == self.opt_arm[self.b]:

			self.opt_arm_val +=1

		#updating the pi function
		




		

	def get_reward(self):
		
		reward = np.random.normal(self.q_star[self.b][self.a],1)
		# print(self.b)
		self.pull_reward.append(reward)
		#Q-value of the action updated

		self.H_val[self.b] -= self.alpha*((reward - self.Q_value[self.b])*( self.pi[self.b]))
		self.H_val[self.b][self.a] += self.alpha*(reward - self.Q_value[self.b][self.a])*(1 - self.pi[self.b][self.a])
		
		self.Q_value[self.b][self.a] = self.Q_value[self.b][self.a] +  (reward - self.Q_value[self.b][self.a])*(1/self.k_i[self.b][self.a])
		

	
	def reset(self):
		
		self.pull_reward=[]
		self.opt_arm_val = 0 

	

	def play(self):
		for self.b in range(self.bandits):
			
			self.select_action_g()

			self.get_reward()
			# print(self.opt_arm_val)



#=====================================================================


k = 10

banditz =2000
pullz = 1000


alpha = [0.1,0.4]
initial = [0,4]
colors = ['r','g','b','k']
fig1=plt.figure().add_subplot(111)
fig2=plt.figure().add_subplot(111)


for alp in alpha:
	print("for " , alp)
	for ini in initial:


		egb = Grad(k,0.1,banditz,alp,ini)
		# print("current eps",ep)
		reward2=[]
		pull_opt =[]


		for i in range(0,pullz):
			

			egb.play()
			reward2.append(np.mean(egb.pull_reward))
			pull_opt.append(float(egb.opt_arm_val)*100/2000)
			
			# print(reward2)

			# epoch_reward += (egb.pull_reward - epoch_reward)/(i+1)
			
			if i%100 ==0:
				# print(egb.actions_taken)
				print("{}/{} pullz done!".format(i,pullz))
				pass
			
			egb.reset()


		fig1.plot(range(pullz),reward2,label = "alpha {} , init {}".format(alp,ini))
		fig2.plot(range(pullz),pull_opt,label = "alpha {} , init {}".format(alp,ini))
			
	
# print(pull_opt)
# print(len(pull_opt))

fig1.set_ylabel('Average Reward')
fig1.set_xlabel('Steps')
# fig1.set_ylim([-0.2,1.6])
fig1.set_xlim([0,1000])
fig1.title.set_text('Average Reward of {} Bandits on each step'.format(banditz))
fig1.legend(loc='best')
fig2.set_ylabel('%optimum actions')
fig2.set_xlabel('Steps')
fig2.title.set_text('%optimum action vs steps')
fig2.legend(loc='best')	

plt.show()

# plt.savefig(cwd +'/opt_action.png',bbox_inches = 'tight')

