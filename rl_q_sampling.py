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
from epidemic import EpidemicEnv





def is_fitted(sklearn_regressor):
	"""Helper function to determine if a regression model from scikit-learn has
	ever been `fit`"""
	return hasattr(sklearn_regressor, 'n_outputs_')

class FQI(object):
	def __init__(self, graph, regressor=None):
		self.simulator = EpidemicEnv(graph=graph)
		self.regressor = regressor or ExtraTreesRegressor()
	
	def state_action(self, states,actions):
		output_state=states.copy()
		if len(actions)>0:
			output_state[actions]=0
			
		return output_state
	

	
	def Q(self, states, actions):
		states, actions = np.array(states), np.array(actions)
		if not is_fitted(self.regressor):
			return np.zeros(len(states))
		else:
			X = np.array([self.state_action(state , action ) for (state,action) in zip(states,actions)])
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

class Memory:
	def __init__(self, state, action, reward, next_state):
		self.state = state
		self.action = action
		self.reward = reward
		self.next_state = next_state

class Memory_belief:
	def __init__(self, state):
		self.state = state


class DQN(FQI):
	def __init__(self, graph, lr=0.005):
		FQI.__init__(self, graph)
		self.feature_size = 2 
		self.best_net = NaiveGCN(node_feature_size=self.feature_size)
		self.net = NaiveGCN(node_feature_size=self.feature_size)
		self.net_list=[] 
		for i in range(int(self.simulator.budget)): 
			self.net_list.append(NaiveGCN(node_feature_size=self.feature_size))
		self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
		self.edge_index = torch.Tensor(list(nx.DiGraph(graph).edges())).long().t()
		self.loss_fn = nn.MSELoss()
		self.replay_memory = []
		self.optimizer_list=[]
		self.replay_memory_list = []
		self.q = .1
		for i in range(int(self.simulator.budget)): 
			self.optimizer_list.append(optim.Adam(self.net_list[i].parameters(), lr=lr))
			self.replay_memory_list.append([])
		self.memory_size = 5000

	def predict_rewards(self, state, action, netid='primary'): 
		features = np.concatenate([[self.state_action(state, action)],[state]], axis=0).T
		net = self.net if netid == 'primary' else self.net_list[netid]
		return net(torch.Tensor(features), self.edge_index) 


	def Q_GCN(self, state, action, netid='primary'):
		node_pred = self.predict_rewards(state, action, netid)
		y_pred = sum(node_pred[action]) 
		return y_pred

	def greedy_action_GCN(self, state):
		action=[]
		possible_actions = self.simulator.possible_nodes.copy()
		for i in range(int(self.simulator.budget)): 
			node_rewards = self.predict_rewards(state, action, netid=i).reshape(-1)
			if len(possible_actions)<2:
				possible_actions=self.simulator.all_nodes.copy()   
			max_indices = node_rewards[possible_actions].argsort()[-1:]
			node=np.array(possible_actions)[max_indices]
			action.append(node)
			possible_actions.remove(node)
			state[node]=0
		return action

	def memory_loss(self, batch_memory, netid='primary', discount=0.98):
		loss_list = []
		for memory in batch_memory:
			state, action, reward, next_state = memory.state.copy(), memory.action.copy(), memory.reward, memory.next_state.copy()
			next_action = self.greedy_action_GCN(next_state)
			prediction = self.Q_GCN(state, action, netid=netid)
			target = reward + discount * self.Q_GCN(next_state, next_action, netid= netid)
			loss = (prediction - target) ** 2 #self.loss_fn(prediction, target)
			loss_list.append(loss)
		total_loss = sum(loss_list)
		return loss

	def fit_GCN(self, num_iterations=100, num_epochs=100, eps=0.1, discount=0.9):  
		writer = SummaryWriter()
		for epoch in range(num_epochs):
			loss_list = []
			cumulative_reward_list = []
			true_cumulative_reward_list = [] 
			true_RL_reward=[]
			self.simulator.Temperature=0 
			for episode in range(num_iterations):
				S, A, R, cumulative_reward = self.run_episode_GCN(eps=eps, discount=discount)
				new_memory = []
				new_memory_list=[]
				for i in range(int(self.simulator.budget)):
					new_memory_list.append([])
				horizon = len(S) - 1
				for i in range(horizon):
					new_memory.append(Memory(S[i], A[i], R[i], S[i+1]))
					act=[]
					for j in range(int(self.simulator.budget)):
						sta=self.state_action(S[i],act)
						act.append(A[i][j])
						rew=float(self.predict_rewards(sta, act, netid='primary')[0])
						new_memory_list[j].append(Memory(sta ,[np.array(A[i][j])],rew ,self.state_action(sta,act)) )

				self.replay_memory += new_memory
				for i in range(int(self.simulator.budget)):
						self.replay_memory_list[i]+=new_memory_list[i]
				# batch_memory = np.random.choice(self.replay_memory, horizon)
				batch_memory=self.replay_memory[-horizon:].copy()
				self.optimizer.zero_grad()
				loss = self.memory_loss(batch_memory, discount=discount)
				loss_list.append(loss.item())
				writer.add_scalar('\\Train/Loss\\', loss.item(), epoch)
				loss.backward()
				self.optimizer.step()
				if len(self.replay_memory) > self.memory_size:
					self.replay_memory = self.replay_memory[-self.memory_size:]
				for i in range(int(self.simulator.budget)):
					# batch_memory=self.replay_memory[-horizon:].copy()
					batch_memory=self.replay_memory_list[i][-horizon:].copy()
					self.optimizer_list[i].zero_grad()
					loss = self.memory_loss(batch_memory,netid=i, discount=discount)
					loss.backward()
					self.optimizer_list[i].step()

				cumulative_reward_list.append(cumulative_reward)
				_, _, true_R, true_cumulative_reward = self.run_episode_GCN(eps=0, discount=1)
				true_RL_reward.append(sum(true_R))
				true_cumulative_reward_list.append(true_cumulative_reward)
			print('Epoch {}, MSE loss: {}, average reward: {}, true RL reward: {}, true reward: {}'.format(epoch, np.mean(loss_list), np.mean(cumulative_reward_list),np.mean(true_RL_reward), np.mean(true_cumulative_reward_list)))
			self.q = max(self.q += .05, 1)
		return cumulative_reward_list,true_cumulative_reward_list
	
	
	def run_episode_GCN(self, eps=0.1, discount=0.99):
		S, A, R = [], [], []
		cumulative_reward = 0
		a=0
		self.simulator.reset(samples=True, q=self.q)
		for t in range(self.simulator.T-1): 
			state = self.simulator.belief_state.copy()
			S.append(state)
			action = self.policy_GCN(state, eps)
			true_state, state_, reward, true_reward = self.simulator.perform(action, t, final_iteration=False)
			A.append(action)
			R.append(reward)
			cumulative_reward += reward * (discount**t)
		state = self.simulator.belief_state.copy()
		S.append(state)
		action = self.policy_GCN(state, eps)
		true_state, state_, reward, true_reward = self.simulator.perform(action, self.simulator.T, final_iteration=True)
		A.append(action)
		R.append(reward)
		cumulative_reward += reward * (discount**t)

		return S, A, R, cumulative_reward
	
	def policy_GCN(self, state, eps=0.1):
		s=state.copy()
		if np.random.rand() < eps:
			return self.random_action()
		else:
			return self.greedy_action_GCN(s) 



def get_graph(graph_index):
	graph_list = ['test_graph','Hospital','India','Exhibition','Flu','irvine','Escorts','Epinions']
	graph_name = graph_list[graph_index]
	path = 'graph_data/' + graph_name + '.txt'
	G = nx.read_edgelist(path, nodetype=int)
	print(G.number_of_nodes())
	mapping = dict(zip(G.nodes(),range(len(G))))
	g = nx.relabel_nodes(G,mapping)
	return g, graph_name

if __name__ == '__main__':
	print('Here goes nothing')
	
	discount=1
	First_time=True
	graph_index=5
	g, graph_name=get_graph(graph_index)
	if First_time:
		model=DQN(graph=g)
		cumulative_reward_list,true_cumulative_reward_list=model.fit_GCN(num_iterations=50, num_epochs=30)
		with open('Graph={}.pickle'.format(graph_name), 'wb') as f:
			pickle.dump([model,true_cumulative_reward_list], f)
	else:
		with open('Graph={}.pickle'.format(graph_name), 'rb') as f:
			X = pickle.load(f) 
		model=X[0]
		true_cumulative_reward_list=X[1]
	cumulative_rewards = []
	for i in range(300):
		model.simulator.Temperature=0
		S, A, R, cumulative_reward = model.run_episode_GCN(eps=0, discount=discount)
		cumulative_rewards.append(np.sum(R))
	print('optimal reward:', np.mean(cumulative_rewards))
	print('optimal reward std:', np.std(cumulative_rewards))