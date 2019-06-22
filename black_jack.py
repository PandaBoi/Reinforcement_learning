import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import gym
import sys
sys.path.append("../")
from modified_env import Black_jack
from collections import defaultdict
 
#with inspiration from https://github.com/dennybritz/reinforcement-learning/
class Agent():

	def __init__(self,env,discount = 1.0,epsilon = 0.1,episodes = 10000):

		self.env = env
		self.discount = discount
		self.actions = [0,1]
		self.epsilon = epsilon
		self.episodes = episodes
		self.state_count = defaultdict(float)
		self.sa_values = defaultdict(float)
		self.state_sum = defaultdict(float)
		self.State_values = defaultdict(float)

	def player_policy(self,state,first_time = True):

		if first_time:
			sum_hand , dealer_hand, ace = state
			return 0 if sum_hand >= 20 else 1
		else:
			# epsilon- greedy policy

			# num = np.random.random()
			# if(num <self.epsilon):
			# 	action = np.random.choice(self.actions, p = [0.5,0.5])
			# else:
			# 	q_vals = [self.sa_values[(state,x)] for x in self.actions]
			# 	action = np.argmax(q_vals)

			# epsilon-soft policy
			actions = np.ones(2) * (self.epsilon/self.env.action_space.n)
			q_vals = [self.sa_values[(state,x)] for x in self.actions]
			best_action = np.argmax(q_vals)
			actions[best_action] += 1 - self.epsilon
			action = np.random.choice(self.actions,p = actions)
			return action


	
	

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
					self.state_count[state_p] =0
				
				self.state_count[state_p] +=1
				self.state_sum[state_p] +=G
				self.State_values[state_p] += (G - self.State_values[state_p])/(self.state_count[state_p] )


	def MC_control(self):
		first_time = True
		for i in range(self.episodes+1):

			if i%1000 ==0:
				print("episode {}/{} done".format(i,self.episodes))

			ep = []
			env = self.env
			state = env.reset()
			done = False
			
			while not done:
				action = self.player_policy(state,first_time)
				next_state , reward , done,_ = env.step(action)
				#appending action as it is needed
				#for estimation of  the state_action values
				ep.append((state,action,reward))
				state = next_state
			first_time = False

			ep_states_action = set([(tuple(x[0]),x[1]) for x in ep])
			# G =0
			for state,action in ep_states_action:
				st_ac = (state,action)
				first_visit = [i for i,q in enumerate(ep) if q[0]==state and q[1] == action][0]
				G = sum([re[2] *(self.discount** i) for i,re in enumerate(ep[first_visit:])])

				if st_ac not in self.sa_values.keys():
					self.sa_values[st_ac] = 0
					self.state_count[st_ac] =0
				
				self.state_count[st_ac] +=1
				self.state_sum[st_ac] +=G
				self.sa_values[st_ac] += (G - self.sa_values[st_ac])/(self.state_count[st_ac] )


















env = Black_jack.BlackjackEnv()
steps = 100000
test = Agent(env,episodes = steps, epsilon = 0.1)

# print(test.sa_values)

#for estimation
# test.estimation_step()
# noace = list(filter(lambda x: (x[2]== False) , test.State_values.keys() ))
# ace = list(filter(lambda x: (x[2]== True) , test.State_values.keys() ))



# noace_val = [(test.State_values[x]) for x in noace]
# ace_val = [(test.State_values[x]) for x in ace]
# keys =list(test.State_values.keys())

# x1 = min(key[0] for key in keys)
# x2 = max(key[0] for key in keys)
# y1 = min(key[1] for key in keys)
# y2 = max(key[1] for key in keys)
# x_a = range(x1,x2+1)
# y_a = range(y1,y2+1)
# X,Y= np.meshgrid(x_a,y_a)
# Z_noace = np.apply_along_axis(lambda _: test.sa_values[(_[0], _[1], False)], 2, np.dstack([X, Y]))
# Z_ace = np.apply_along_axis(lambda _: test.sa_values[(_[0], _[1], True)], 2, np.dstack([X, Y]))


#for control

test.MC_control()
# print(test.sa_values)
noace = list(filter(lambda x: (x[0][2]== False) , test.sa_values.keys() ))
ace = list(filter(lambda x: (x[0][2]== True) , test.sa_values.keys() ))


noace_val = [(test.sa_values[x]) for x in noace]
ace_val = [(test.sa_values[x]) for x in ace]
keys =list(test.sa_values.keys())
# print(ace)

x1 = min(key[0][0] for key in keys)
x2 = max(key[0][0] for key in keys)
y1 = min(key[0][1] for key in keys)
y2 = max(key[0][1] for key in keys)
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

Z_noace = np.apply_along_axis(lambda _: test.sa_values[(_[0], _[1], False),1], 2, np.dstack([X, Y]))
Z_noace = np.apply_along_axis(lambda _: test.sa_values[(_[0], _[1], False),0], 2, np.dstack([X, Y]))
Z_ace = np.apply_along_axis(lambda _: test.sa_values[(_[0], _[1], True),1], 2, np.dstack([X, Y]))
Z_ace = np.apply_along_axis(lambda _: test.sa_values[(_[0], _[1], True),0], 2, np.dstack([X, Y]))

# print(len(Y))

make_graph_3d(X, Y, Z_noace,'{} (no usable ace)'.format(steps))
make_graph_3d(X, Y, Z_ace,'{} (usable ace)'.format(steps))