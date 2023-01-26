import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
import pickle
import time
import random
import networkx as nx
from tqdm import tqdm, trange

from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv, GraphConv, SAGEConv, GatedGraphConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

from torch.utils.tensorboard import SummaryWriter     

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesRegressor

from gcn import NaiveGCN
from epidemic_periodic import EpidemicEnv

def is_fitted(sklearn_regressor):
	"""Helper function to determine if a regression model from scikit-learn has
	ever been `fit`"""
	return hasattr(sklearn_regressor, 'n_outputs_')

class FQI(object):
	def __init__(self, graph, regressor=None):
		self.simulator = EpidemicEnv(graph=graph)
		self.regressor = regressor or ExtraTreesRegressor()
	
	def state_action(self, states,actions):
		output_state = states.copy()
		if len(actions) > 0:
			output_state[actions] = 0
			
		return output_state
	

	def Q(self, states, actions):
		states, actions = np.array(states), np.array(actions)
		if not is_fitted(self.regressor):
			return np.zeros(len(states))
		else:
			X = np.array([self.state_action(state, action) for (state, action) in zip(states, actions)])
			y_pred = self.regressor.predict(X)
			return y_pred    
	
	def greedy_action(self, state):
		action = []
		possible_actions = self.simulator.possible_nodes
		if len(possible_actions)>int(self.simulator.budget):
			np.random.shuffle(possible_actions)
			Q_values = self.Q([state]*len(possible_actions), [[j] for j in possible_actions]) # enumerate all the possible nodes
			index=Q_values.argsort()[-int(self.simulator.budget):]
			next_action=[possible_actions[v] for v in index]
		else:
			next_action=np.array(possible_actions)
		return list(next_action)   
	
	def random_action(self):
		if len(self.simulator.possible_nodes)>0:
			action = random.sample(self.simulator.possible_nodes,int(min(len(self.simulator.possible_nodes),self.simulator.budget)))
		else:
			action = random.sample(self.simulator.all_nodes,int(min(self.simulator.n,self.simulator.budget)))
		return action
	
	def policy(self, state, eps=0.1):
		if np.random.rand() < eps:
			return self.random_action()
		else:
			return self.greedy_action(state) 
	
	def run_episode(self, eps=0.1, discount=0.98):
		S, A, R = [], [], []
		cumulative_reward = 0
		self.simulator.reset()
		state = self.simulator.state
		for t in range(self.simulator.T):    
			state = self.simulator.state
			S.append(state)
			action = self.policy(state, eps)
			state_, reward=self.simulator.step(action=action)#Transition Happen 
			state=state_
			A.append(action)
			R.append(reward)
			cumulative_reward += reward * (discount**t)
		return S, A, R, cumulative_reward


	def fit_Q(self, episodes, num_iters=10, discount=0.9):
		prev_S = []
		next_S = []
		rewards = []
		for (S, A, R) in episodes:
			horizon = len(S)
			for i in range(horizon-1):
				prev_S.append(list(self.state_action(S[i], A[i])) )
				rewards.append(R[i])
				next_S.append(S[i+1])
				

		prev_S = np.array(prev_S)
		next_S = np.array(next_S)
		rewards = np.array(rewards)

		for iteration in range(num_iters):
			best_actions = [self.greedy_action(state) for state in next_S]
			Q_best = self.Q(next_S, best_actions)
			y = list(rewards + discount * np.array(Q_best))
			self.regressor.fit(prev_S, y)

	
	def fit(self, num_refits=10, num_episodes=10, discount=0.9, eps=0.1):
		cumulative_rewards = np.zeros((num_refits, num_episodes))
		for refit_iter in range(num_refits):
			episodes = []
			for episode_iter in range(num_episodes):
				S, A, R, cumulative_reward = self.run_episode(eps=eps, discount=discount)
				cumulative_rewards[refit_iter,episode_iter] = cumulative_reward
				episodes.append((S, A, R))
			print('average reward:', np.mean(cumulative_rewards[refit_iter]))
			self.fit_Q(episodes,discount=discount)

		return episodes, cumulative_rewards    