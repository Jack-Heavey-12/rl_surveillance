from epidemic import EpidemicEnv

import numpy as np
import random
import networkx as nx


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
		future_tests = set()
		for i in range(100):
			if not list(future_tests): 
				action = random.sample(env.all_nodes,min(len(env.all_nodes),env.budget))
	
			elif len(list(future_tests)) >= env.budget:	
				#Contact Tracing List
				deg_lst = sorted(g.degree, key=lambda x: x[1], reverse=True)
				action = [x for x in deg_lst if x in future_tests][:env.budget] #sorted(G.degree, key=lambda x: x[1], reverse=True)[:env.budget]
			else:
				#don't need to sort here because all neighbors will be tested regardless
				num = env.budget - len(list(future_tests)) 
				action = random.sample(env.all_nodes,num) + list(future_tests)

			reward = env.perform(action, i, final_iteration=False)

			positive_tests = [ind for ind, x in enumerate(reward[1]) if x == 1]
			
			future_tests = set()
			for j in positive_tests:
				future_tests.update(g.neighbors(j))
			


		if not list(future_tests): 
			action = random.sample(env.all_nodes,min(len(env.all_nodes),env.budget))
	
		elif len(list(future_tests)) >= env.budget:	
			#Contact Tracing List
			action = random.sample(list(future_tests), env.budget)
		else:
			num = env.budget - len(list(future_tests)) 
			action = random.sample(env.all_nodes,num) + list(future_tests)
		reward = env.perform(action, i, final_iteration=True)
		total_rew.append(env.terminal_reward)
	print('=== Contact Tracing Degree Sorting ===')
	print(f'Reward Average: {np.mean(total_rew)}, Reward Standard Deviation: {np.std(total_rew)}')