import numpy as np
import matplotlib.pyplot as plt

'''

Gambler's Problem - Chapter 4 Dynamic Programming

	GOAL = 100
	GAME_OVER = 0
	State_values = the possible states [0,100]
	actions = possible actions (0,min(s,100-s))
	ph = prob of heads = {0.25,0.4,0.55}
	discount = 1

	
'''

class Gambler():

	def __init__(self,ph):

		self.ph = ph
		self.State_values = np.zeros(101)
		self.State_values[0],self.State_values[100] = 0,1
		self.policy = np.zeros((101),dtype = np.int)
		self.discount = 1
		self.GOAL = 100
		self.GAME_OVER = 0


	def q_value(self,state,action,State_vals):

		q_val = self.ph*(State_vals[state+action]) + (1-self.ph)*(State_vals[state-action])
		# print(q_val)
		return q_val

	def play(self):

		while(True):
			error =0.0
			v_copy = np.copy(self.State_values)
			for s in range(1,100):
				
				actions = range(min(s,100-s)+1)
				q_values =[]
				for a in actions:
					q_values.append(self.q_value(s,a,self.State_values))

				v_copy[s] = np.max(np.array(q_values))
				
				

			error = np.abs(v_copy - self.State_values).sum()
			# print("v changed by",error)
			self.State_values = v_copy
			if(error <1e-10):
				print("converged")
				print(self.State_values)
				break

		for s in range(1,100):
			actions = np.arange(0,min(s,100-s)+1)
			q_values =[]
			for a in actions:
				q_values.append(self.q_value(s,a,self.State_values))

			self.policy[s] = actions[np.argmax(np.round(q_values[1:],5)) +1 ]

		print("policy extracted")


ph = [0.25,0.4,0.5]
colors = ['y','g','b']
fig1=plt.figure().add_subplot(111)
fig2=plt.figure().add_subplot(111)

for p in ph:
	test = Gambler(p)
	test.play()
	fig1.plot(range(101),test.State_values,label = "Prob of head {}".format(p))
	fig2.plot(range(101),test.policy,label = "Prob of head{}".format(p))

fig1.set_ylabel('State_value')
fig1.set_xlabel('Capital')
fig1.legend(loc='best')
fig2.set_ylabel('Stakes')
fig2.set_xlabel('Capital')
fig2.legend(loc='best')	

plt.show()

# plt.savefig(cwd +'/opt_action.png',bbox_inches = 'tight')





