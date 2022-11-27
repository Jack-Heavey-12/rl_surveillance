import numpy as np
import networkx as nx

def get_graph(graph_index):
    graph_list = ['test_graph'] #['test_graph','Hospital','India','Exhibition','Flu','irvine','Escorts','Epinions']
    graph_name = graph_list[graph_index]
    path = f'graph_data/{graph_name}.txt'
    G = nx.read_edgelist(path, nodetype=int)
    print(G.number_of_nodes())
    mapping = dict(zip(G.nodes(),range(len(G))))
    g = nx.relabel_nodes(G,mapping)
    return g, graph_name


def environment_update(G, iso_t):
	#Will update the environment to the next time step, giving x_{t+1} from x_t