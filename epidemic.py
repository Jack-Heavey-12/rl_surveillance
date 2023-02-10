# File that defines the epidemic environment class
# Based off of SIS_Belief_env.py

import numpy as np
import random
import networkx as nx
from matplotlib import pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor


class EpidemicEnv(object):
	def __init__(self, graph, budget_c = None, initial_i = 0.1, infect_prob=0.1, cure_prob=0.05, iso_length = 10, tau = 1, intermediate_sample = .1):
		self.graph = graph
		self.n = len(graph)
		if not budget_c:
			self.budget = int(self.n / 10)
		else:
			self.budget = budget_c

		self.sample = int(intermediate_sample * self.budget)
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
		#Not sure what this variable does? I think it's for the curriculum learning?
		self.Temperature = 1
		self.belief_regressor=ExtraTreesRegressor()
		#dictionary of infected individuals at a given time - each dictionary value should be a set.
		self.inf_t = {}
		#running list of infected individuals based on the test
		self.infections = []
		#self.inf_t[0] = set()
		#For isolation, we will have a set of people currently in isolation, a dict for how much time in isolation left,
		# and a list for calculating spreading infections - ones if they are not in isolation and zeroes if they are
		self.iso = set()
		self.iso_time = {}
		self.inv_iso = np.ones(self.n)

	#For resetting the environment for multiple trials
	def reset(self):
		self.all_nodes = list(range(self.n))
		self.true_state = np.zeros(self.n)
		self.true_state[random.sample(self.all_nodes, int(self.n*self.Initial_I))] = 1
		self.belief_state = self.Initial_I * np.ones(self.n)
		self.A = nx.to_numpy_matrix(self.graph)

		self.inf_t = {}
		self.iso = set()
		self.iso_time = {}
		self.inv_iso = np.ones(self.n)

		

	def perform(self, iso_list, t, final_iteration = False):
		S_true = self.true_state.copy()
		S_belief = self.belief_state.copy()

		if not self.is_action_legal(iso_list):
			print("Exceed Budget")
			iso_list = []
			


		#Can do some curriculum learning here.
		if not final_iteration:
			hidden_reward = -np.sum(self.true_state)
			#visible_reward = 0
			lst = self.all_nodes.copy()
			rand_sample = random.sample(lst, self.sample)
			visible_reward = sum([self.true_state[v] for v in rand_sample]) * -1

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

		detected_infections = []
		detected_negatives = []
		for i in iso_list:
			if self.true_state[i] == 1:
				detected_infections.append(i)
				if i not in self.iso:
					self.iso.add(i)
					self.iso_time[i] = self.iso_length
			else:
				detected_negatives.append(i)

		#used in the belief_transition update
		self.infections = detected_infections
		self.negatives = detected_negatives
		# Moving the infection state one step forward.
		next_cure_list = [v for v in self.all_nodes if self.true_state[v] == 1 and random.uniform(0,1) < self.cure_prob]

		next_infect_list = [v for v in self.all_nodes if self.true_state[v] == 0 and \
									random.uniform(0,1) < 1 - (1 - self.infect_prob)**(np.inner(self.true_state * self.inv_iso, self.A[v].A1))]

		#Updates the true states for people that get cured/infected
		for i in next_cure_list:
			self.true_state[i] = 0
		for i in next_infect_list:
			self.true_state[i] = 1

		next_belief_state = self.belief_transition(detected_infections, detected_negatives, self.belief_state)
		self.belief_state = next_belief_state



		return self.true_state, self.belief_state, total_reward, min(total_reward, np.sum(np.array(self.true_state) * -1))


	def belief_transition(self, known_infections, known_negatives, previous_belief):
		
		previous_belief[known_infections] = 1
		previous_belief[known_negatives] = 0

		belief_state = np.array(previous_belief)
		
		indegree_prob = np.ones(self.n) - self.infect_prob * previous_belief
		#added the additional requirement that an individual isn't in isoluation. Think this is done correctly
		Prob = np.array([np.prod([(1 - indegree_prob[u]) for u in self.A[:,v].nonzero()[0] if (self.inv_iso[u] == 1 or u in known_infections)]) for v in self.all_nodes])

		belief_state = previous_belief + (np.ones(self.n) - previous_belief) * (np.ones(self.n) - Prob) # we know all the reported already thus no (1-c)
	

		#Updates the probability of an individual being infected while they are in isolation. That belief should decrease as isolation continues
		for i in list(self.iso_time.keys()):
			belief_state[i] = (1 - self.cure_prob) ** (self.iso_length - self.iso_time[i])

		belief_state[known_infections] = 1
		belief_state[known_negatives] = 0

		return belief_state

	def is_action_legal(self, iso_list):
		if(len(list(iso_list)) <= self.budget):
			return True
		else:
			return False    

if __name__ == '__main__':
	print('Here goes nothing')
	Graph_List = ['test_graph','Hospital','India','Exhibition','Flu','irvine','Escorts','Epinions']
	Graph_index = 2
	Graph_name = Graph_List[Graph_index]
	path = 'graph_data/' + Graph_name + '.txt'
	G = nx.read_edgelist(path, nodetype=int)
	mapping = dict(zip(G.nodes(),range(len(G))))
	g = nx.relabel_nodes(G,mapping)
	env = EpidemicEnv(graph=g)
	
	total_rew = []
	for _ in range(50):
		env.reset()
		true_I = []
		belief_I = []
		#Reward = []
		for i in range(100):
			
			action = random.sample(env.all_nodes,min(len(env.all_nodes),env.budget))
			reward = env.perform(action, i, final_iteration=False)
			#true_I.append(sum(S1))
			Reward.append(reward[2])
		action = random.sample(env.all_nodes,min(len(env.all_nodes),env.budget))
		reward = env.perform(action, i, final_iteration=True)
		Reward.append(reward[2])
		plt.plot(range(len(true_I)),true_I)
		# plt.plot(range(len(belief_I)),belief_I)
		total_rew.append(sum(Reward))

	print(f'Reward Average: {mean(total_rew)}, ')
