# File that defines the epidemic environment class
# Based off of SIS_Belief_env.py

import numpy as np
import random
import networkx as nx
from matplotlib import pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor


class EpidemicEnv(object):
	def __init__(self, graph, budget_c = 5, initial_i = 0.1, infect_prob=0.1, cure_prob=0.05, iso_length = 10, tau = 1):
		self.graph = graph
		self.n = len(graph)
		self.budget = budget_c
		self.Initial_I = initial_i #random.sample(list(range(self.n)), int(self.n * initial_i))
		self.infect_prob = infect_prob
		self.cure_prob = cure_prob
		self.all_nodes = list(range(self.n))
		self.true_state = np.zeros(self.n)
		self.true_state[random.sample(self.all_nodes, int(self.n*self.Initial_I))] = 1
		self.possible_nodes = list(range(self.n))
		self.iso_length = iso_length
		self.tau = tau

		self.A = nx.to_numpy_matrix(graph)
		#Should the initial belief state be unknown due to an unknown initial infection rate? Something to think about
		self.belief_state = self.Initial_I * np.ones(self.n)
		#time steps
		self.T = 100
		#Not sure what this variable does?
		self.Temperature = 1
		self.belief_regressor=ExtraTreesRegressor()
		#dictionary of infected individuals at a given time - each dictionary value should be a set.
		self.inf_t = {}
		#self.inf_t[0] = set()
		#For isolation, we will have a set of people currently in isolation, a dict for how much time in isolation left,
		# and a list for calculating spreading infections - ones if they are not in isolation and zeroes if they are
		self.iso = set()
		self.iso_time = {}
		self.inv_iso = np.ones(self.n)


	### Need to figure out how to do this - not immediately obvious.
	def perform(self, iso_list, t, final_iteration = False):
		S_true = self.true_state.copy()
		S_belief = self.belief_state.copy()

		if not self.is_action_legal(iso_list):
			print("Exceed Budget")
			iso_list = []
			


		#Can do some curriculum learning here.
		if not final_iteration:
			hidden_reward = -np.sum(self.true_state)
			visible_reward = 0

			total_reward = self.tau * visible_reward + (1 - self.tau) * hidden_reward
		else:
			total_reward = 0
			for i in list(self.inf_t.keys()):
				total_reward += len(self.inf_t[i])
			total_reward = -total_reward


		#Saving the infected set for calculation later
		self.inf_t[t] = set([v for v in self.all_nodes if self.true_state[v] == 1])

		#Moves the isolation time one step forward
		copy = self.iso.copy()
		for i in copy:
			self.iso_time[i] -= 1
			if self.iso_time[i] == 0:
				self.iso.remove(i)
				del self.iso_time[i]
				self.inv_iso[i] = 1

		# Setting up the individuals that go into isolation at a given step.
		for i in iso_list:
			if self.true_state[i] == 1:
				if i not in self.iso:
					self.iso.add(i)
					self.iso_time[i] = self.iso_length

		# Moving the infection state one step forward.
		next_cure_list = [v for v in self.all_nodes if self.true_state[v] == 1 and random.uniform(0,1) < self.cure_prob]

		next_infect_list = [v for v in self.all_nodes if self.true_state[v] == 0 and \
									random.uniform(0,1) < 1 - (1 - self.infect_prob)**(np.inner(self.true_state * self.inv_iso, self.A[v].A1))]

		#Updates the true states for people that get cured/infected
		for i in next_cure_list:
			self.true_state[i] = 0
		for i in next_infect_list:
			self.true_state[i] = 1

		return total_reward
		


	def is_action_legal(self, iso_list):
		if(len(list(iso_list)) <= self.budget):
			return True
		else:
			return False    

if __name__ == '__main__':
	print('Here goes nothing')
	Graph_List = ['test_graph','Hospital','India','Exhibition','Flu','irvine','Escorts','Epinions']
	Graph_index = 0
	Graph_name = Graph_List[Graph_index]
	path = 'graph_data/' + Graph_name + '.txt'
	G = nx.read_edgelist(path, nodetype=int)
	mapping = dict(zip(G.nodes(),range(len(G))))
	g = nx.relabel_nodes(G,mapping)
	env = EpidemicEnv(graph=g, budget_c = 5)
	
	#env.reset()
	true_I = []
	belief_I = []
	Reward = []
	for i in range(100):
		
		action = random.sample(env.all_nodes,min(len(env.all_nodes),env.budget))
		reward = env.perform(action, i, final_iteration=False)
		#true_I.append(sum(S1))
		Reward.append(reward)
	action = random.sample(env.all_nodes,min(len(env.all_nodes),env.budget))
	reward = env.perform(action, i, final_iteration=True)
	Reward.append(reward)
	plt.plot(range(len(true_I)),true_I)
	# plt.plot(range(len(belief_I)),belief_I)
	print(sum(Reward))
