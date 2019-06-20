import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import gym
import sys
sys.path.append("../")
from modified_env import Black_jack
from collections import defaultdict
 

class Agent():

	def __init__(self,env,discount = 1.0,episodes = 10000):

		self.env = env
		self.discount = discount
		self.actions = [0,1]
		self.episodes = episodes
		self.state_count = defaultdict(float)

		self.State_values = defaultdict(float)
		self.State_values2 = defaultdict(float)
		self.state_sum = defaultdict(float)

	def player_policy(self,state):

		sum_hand , dealer_hand, ace = state
		return 0 if sum_hand >= 20 else 1

	
	

	def estimation_step(self):

		for i in range(self.episodes+1):

			if i%1000 ==0:
				print("episode {}/{} done".format(i,self.episodes))

			ep = []
			env = self.env
			state = env.reset()
			done = False
			while not done:
				action = self.player_policy(state)
				next_state , reward , done,_ = env.step(action)
				#not appending action as it is not needed
				#for estimation of just the state values
				ep.append((state,reward))
				state = next_state

			# ep.append((state,reward))

			ep_states = set([tuple(x[0]) for x in ep])
			# G =0
			for state_p in ep_states:
				first_visit = [i for i,q in enumerate(ep) if q[0]==state_p][0]
				G = sum([re[1] *(self.discount** i) for i,re in enumerate(ep[first_visit:])])

				if state_p not in self.State_values.keys():
					self.State_values[state_p] = 0
					self.State_values2[state_p] = 0
					self.state_count[state_p] =0
					self.state_sum[state_p] = 0
				
				self.state_count[state_p] +=1
				self.state_sum[state_p] +=G
				self.State_values[state_p] += (G - self.State_values[state_p])/(self.state_count[state_p] )
				self.State_values2[state_p] = (self.state_sum[state_p])/(self.state_count[state_p])


env = Black_jack.BlackjackEnv()
steps = 50000
test = Agent(env,episodes = steps)
test.estimation_step()
noace = list(filter(lambda x: (x[2]== False) , test.State_values.keys() ))
ace = list(filter(lambda x: (x[2]== True) , test.State_values.keys() ))

noace_val = [(test.State_values[x]) for x in noace]
ace_val = [(test.State_values[x]) for x in ace]
keys =list(test.State_values.keys())

x1 = min(key[0] for key in keys)
x2 = max(key[0] for key in keys)
y1 = min(key[1] for key in keys)
y2 = max(key[1] for key in keys)
x_a = range(x1,x2+1)
y_a = range(y1,y2+1)
X,Y= np.meshgrid(x_a,y_a)


def make_graph_3d(X,Y,Z,title):
	fig1 = plt.figure(figsize=(20,10))
	ax = fig1.add_subplot(111,projection = '3d')
	surface = ax.plot_surface(X,Y,Z,rstride = 1,cstride =1,cmap = matplotlib.cm.summer)
	ax.set_xlabel('player\'s hand')
	ax.set_ylabel("Dealer's hand")
	ax.set_zlabel("State_values")
	ax.set_title(title)
	plt.show()

noace_val =np.array(noace_val)[:,np.newaxis]
Z_noace = np.apply_along_axis(lambda _: test.State_values[(_[0], _[1], False)], 2, np.dstack([X, Y]))
Z_ace = np.apply_along_axis(lambda _: test.State_values[(_[0], _[1], True)], 2, np.dstack([X, Y]))

print(len(Y))

make_graph_3d(X, Y, Z_noace,'{} (no usable as)'.format(steps))
make_graph_3d(X, Y, Z_ace,'{} (usable as)'.format(steps))