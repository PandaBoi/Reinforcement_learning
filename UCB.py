import numpy as np
import matplotlib.pyplot as plt
import random
import os
import math

cwd = os.getcwd()
# print(cwd)

"""
UCB multi armed bandit problem of Sutton & Barto book


"""
#==========================  UCB MAB ============================
class UCB_MAB:
	def __init__(self,k,c,bandits):

		#initializing all parameters

		self.t = 0
		self.k = k
		self.c = c
		self.bandits = bandits

			

		self.k_i = np.ones((bandits,k)) # no of occ of kth arm for bth bandit
		self.pull_reward = np.zeros(bandits) #avg reward of each step
		self.avg_reward = 0.0
		self.q_star = np.random.normal(0,1,(bandits,k))
		self.Q_value = np.zeros((bandits,k))
		self.opt_arm = np.argmax(self.q_star,1)# the highest reward arms 
		self.actions_taken = []
		
		self.pull_reward=[]
		self.opt_arm_pulled_tot=[]
		self.opt_arm_val =0

	def select_action_c(self):
		self.a  = np.argmax(self.Q_value[self.b] + self.c * (np.sqrt( (np.log(self.t)) / (self.k_i[self.b]) )))
		self.k_i[self.b][self.a] +=1
		self.actions_taken.append(self.a)

		if self.a == self.opt_arm[self.b]:
			self.opt_arm_val +=1



		

	def get_reward_c(self):
		
		reward = np.random.normal(self.q_star[self.b][self.a],1)
		# print(self.b)
		self.pull_reward.append(reward)	
		#Q-value of the action updated
		self.Q_value[self.b][self.a] = self.Q_value[self.b][self.a] +  (reward - self.Q_value[self.b][self.a])*(1/self.k_i[self.b][self.a])
		
		

	def reset(self):
		
		self.pull_reward=[]
		self.opt_arm_val = 0 

	def play(self):
		self.t += 1
		for self.b in range(self.bandits):
			
			self.select_action_c()

			self.get_reward_c()
			# print(self.opt_arm_val)



#=====================================================================

k = 10

banditz =2000
pullz = 1000


c = [1,2,0]
colors = ['r','g','b','k']
fig1=plt.figure().add_subplot(111)
fig2=plt.figure().add_subplot(111)



for _,cc in enumerate(c):

	egb = UCB_MAB(k,cc,banditz)
	print("current c",cc)
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


	fig1.plot(range(pullz),reward2,colors[_],label='c_value = {}'.format(cc))
	fig2.plot(range(pullz),pull_opt,colors[_],label = 'c_value = {}'.format(cc))
			
	
# print(pull_opt)
# print(len(pull_opt))

fig1.set_ylabel('Average Reward')
fig1.set_xlabel('Steps')
fig1.set_ylim([-0.2,1.6])
fig1.set_xlim([0,1000])
fig1.title.set_text('Average Reward of {} Bandits on each step'.format(banditz))
fig1.legend(loc='best')
fig2.set_ylabel('%optimum actions')
fig2.set_xlabel('Steps')
fig2.set_ylim([0,100])
fig2.title.set_text('%optimum action vs steps')
fig2.legend(loc='best')	

plt.show()

# plt.savefig(cwd +'/opt_action.png',bbox_inches = 'tight')

