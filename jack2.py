import numpy as np
import matplotlib.pyplot as plt
import pickle
from math import exp,factorial


LAMBDA_RET_1 = 3
LAMBDA_RET_2 = 2
LAMBDA_REQ_1 = 3
LAMBDA_REQ_2 = 4
INCOME = 10
PENALTY = 2
DISCOUNT = 0.9
MAX_CARS = 20
MOV_CARS = 5

actions = np.arange(- MOV_CARS,  MOV_CARS +1) # (-5,5)

poisson_cache = dict()

def poisson_calc( n,lam):

	global poisson_cache
	# # used from - https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter04/car_rental.py
	key = n * 10 + lam
	if key not in  poisson_cache.keys():
		 poisson_cache[key] = exp(-lam) * pow(lam, n) / factorial(n)
	return  poisson_cache[key]

	# res = exp(-lam) * pow(lam, n) / factorial(n)

	# return res




def step(state,a,sv):

	s = 0.0

	s -= PENALTY * abs(a) 

	#range actually goes from 0 -20 but for faster computation 10 is uppercap
	for req_1 in range(0,11):
		for req_2 in range(0,11):
			
			cars_av_1 = int(min(state[0] - a,  MAX_CARS))
			cars_av_2 = int(min(state[1]+ a,  MAX_CARS))


			av_to_give1 = min(cars_av_1,req_1)
			av_to_give2 = min(cars_av_2,req_2)

			reward = (av_to_give2 + av_to_give1) *   INCOME

			cars_av_1 -= av_to_give1
			cars_av_2 -= av_to_give2
			prob = poisson_calc(req_1, LAMBDA_REQ_1) * poisson_calc(req_2, LAMBDA_REQ_2)
			# print(prob)

			for ret_1 in range(0,11):
				for ret_2 in range(0,11):
			
					cars_av_1_f = min(cars_av_1 + ret_1, MAX_CARS)
					cars_av_2_f = min(cars_av_2 + ret_2,  MAX_CARS)

					prob_t =  poisson_calc(ret_1, LAMBDA_RET_1) * poisson_calc(ret_2, LAMBDA_RET_2) * prob
					# print(prob)		 
					s += prob_t * (reward + DISCOUNT * sv[cars_av_1_f, cars_av_2_f])
	# print(s)
	return s

	

def play():
	
	State_values = np.zeros(( MAX_CARS +1, MAX_CARS +1))
	policy = np.zeros(State_values.shape,dtype = np.int)
	
	while(1):
		ep =0
	# Using Policy iteration - Step 1. policy evaluation
		while(1):
			values_co =  np.copy( State_values)
			for i in range(MAX_CARS+1):
				for j in range(MAX_CARS+1):
					values_co[i,j] =  step([i,j], policy[i,j],values_co)
					# print(i,j,values_co[i,j])
			change =np.abs((values_co -  State_values)).sum()
			print("value changed by %f"  %change )
			
			State_values = values_co

			if change< 1e-4:
				print("policy evaluated")
				break
			
	#Step 2. policy improvement

		policy_co = np.copy( policy)

		for i in range( MAX_CARS+1):
			for j in range( MAX_CARS+1):

				qvalues= []
				for a in  actions:
					if (a >=0 and a <= i) or (a <0 and abs(a) <=j):
						qvalues.append( step([i,j],a, State_values))
					else:
						qvalues.append(-float('inf'))
 
				policy_co[i,j] = actions[np.argmax(qvalues)]
	
		policy_diff = (policy_co !=  policy).sum()
		ep +=1
		print("diff in policy",policy_diff)		
		policy = policy_co

		name = 'policy__'+str(ep)+'.pkl'
		f = open(name,'wb')
		pickle.dump(policy,f)
		f.close()
		
		if policy_diff == 0:
			print("policy improved")
			break
		
	
	

	f = open('state_val.pkl','wb')
	pickle.dump(State_values,f)
	f.close()
	
	

if __name__ == '__main__':
	play()
	
	import pickle
import numpy as np
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	f = open('state_val.pkl','rb')
	sv = pickle.load(f)
	f.close()
	print(sv.shape)

	x = np.arange(0,21)
	y = np.arange(0,21)
	z = sv
	fig = plt.figure()
	x,y = np.meshgrid(x,y)
	ax = Axes3D(fig)
	ax.plot_surface(x,y,z,cmap ='hot')
	ax.set_xlabel('# of cars at 1')
	ax.set_ylabel('# of cars at 2')
	ax.set_zlabel('State_values')
	plt.show()