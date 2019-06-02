import numpy as np
import matplotlib.pyplot as plt








class GRid_World:


	def __init__ (self,size = 5,A = [0,1],A_ = [4,1],B = [0,3],B_ =  [2,3],gamma = 0.9):

		self.size = size
		self.A = np.array(A)
		self.A_ = np.array(A_)
		self.B = np.array(B)
		self.B_ = np.array(B_)
		self.gamma = gamma
		self.prob = np.array([0.25,0.25,0.25,0.25])

		# actions up,down,left,right
		self.actions  = np.array([[-1,0],[1,0],[0,-1],[0,1]])
		self.V_values = np.zeros((self.size,self.size))
		self.reward = 0.0
		self.actions_taken =[]


	


	def take_step(self,state):
		
		next_state = []
		reward = 0
		state = np.array(state)[np.newaxis,:]

		if np.all(state == self.A):

			next_state = np.repeat(self.A_[np.newaxis,:],4,0)
			reward = np.repeat([[10]],4,0)

			

		elif np.all(state == self.B):


			next_state = np.repeat(self.B_[np.newaxis,:],4,0)
			reward = np.repeat([[5]],4,0)

			

		else:
			next_state = np.array([[100,100]])
			reward = np.array([[100]])
			for _,a in enumerate(self.actions):
				
				temp = a + state
				
				if np.all(temp[0][0]< 0) or np.all(temp[0][0] >=self.size) or np.all(temp[0][1]< 0) or np.all(temp[0][1] >=self.size): 
					
					next_state = np.append(next_state,state,0)
					reward = np.append(reward,[[-1]],0)
				
				else:

					next_state = np.append(next_state,temp,0)
					reward = np.append(reward,[[0]],0)
			
			

			next_state = next_state[1:,:]
			reward = reward[1:,:]

		return next_state, reward


	def bellman_compute(self):

		while True:

			temp_V_value = np.zeros((self.size,self.size))

			for i in range(self.size):
				for j in range(self.size):

					next_state , reward = self.take_step([i,j])
					v_next = np.array([[ self.V_values[s[0]][s[1]] ] for s in next_state])
					
					a = reward +self.gamma*v_next
					b = np.squeeze(a)
					temp_V_value[i][j] = b.dot(self.prob)

			if np.sum(np.abs(temp_V_value - self.V_values)) < 1e-4:
				break
			self.V_values = temp_V_value

	def bellman_opt_compute(self):

		while True:

			temp_V_value = np.zeros((self.size,self.size))

			for i in range(self.size):
				for j in range(self.size):

					next_state , reward = self.take_step([i,j])
					v_next = np.array([[ self.V_values[s[0]][s[1]] ] for s in next_state])
					
					a = reward +self.gamma*v_next
					b = np.squeeze(a)
					temp_V_value[i][j] = np.max(b)

			if np.sum(np.abs(temp_V_value - self.V_values)) < 1e-4:
				break
			self.V_values = temp_V_value

test = GRid_World()

# next_s ,reward=test.take_step([0,1])
# print(next_s[0][1])
test.bellman_opt_compute()
print(np.around(test.V_values, decimals = 1))
x = np.around(test.V_values, decimals = 1)
fig1=plt.figure().add_subplot(111)
fig1.axis('off')

colLables = ('1','2','3','4','5')
rowLables= ('1','2','3','4','5')
table = fig1.table(x,loc = 'center',colLabels = colLables,
					rowLabels = rowLables)
plt.show()


















# make the grid


# make the game

#  graphing the progress