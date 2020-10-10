import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

evaluated=0

assert(nx.__version__ == '2.2' or nx.__version__ == '2.1')

def load_graph(filename):
    assert os.path.exists(filename)
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        print('Loaded graph from {}'.format(f))
    return data['G']

def make_graph(env, sampler, connection_radius, num_vertices, directed=True, lazy=False, saveto='graph.pkl'):
    """
    Returns a graph ont he passed environment.
    All vertices in the graph must be collision-free.

    Graph should have node attribute "config" which keeps a configuration in tuple.
    E.g., for adding vertex "0" with configuration np.array([0, 1]),
    G.add_node(0, config=tuple(config))

    To add edges to the graph, call
    G.add_weighted_edges_from(edges)
    where edges is a list of tuples (node_i, node_j, weight),
    where weight is the distance between the two nodes.

    @param env: Map Environment for graph to be made on
    @param sampler: Sampler to sample configurations in the environment
    @param connection_radius: Maximum distance to connect vertices
    @param num_vertices: Minimum number of vertices in the graph.
    @param lazy: If true, edges are made without checking collision.
    @param saveto: File to save graph and the configurations
    """
    if directed:
        # Used for Dubins path.
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    # TODO: Implement here
    # 1. Sample vertices, check state validity.
    # 2. Connect them with edges (support lazy=True)
    # When adding edges, note that the order of nodes matter for directed graphs,
    # i.e., (node_i, node_j, weight) != (node_j, node_i, weight).

    vertices = sampler.sample(num_vertices)
    # count evaluated edges (for write up)
    global evaluated    
    for config in vertices:
	add_node(G, config, env, connection_radius, directed, lazy) 
    # Save the graph to reuse.
    #uncomment
    if saveto is not None:
        data = dict(G=G)
        pickle.dump(data, open(saveto, 'wb'))
        print('Saved the graph to {}'.format(saveto))
    print("Edges evaluated by graph: ",evaluated)
    return G


def add_node(G, config, env, connection_radius, start_from_config, lazy):
    """
    This function should add a node to an existing graph G.
    @param G graph, constructed using make_graph
    @param config Configuration to add to the graph
    @param env Environment on which the graph is constructed
    @param connection_radius Maximum distance to connect vertices
    @param start_from_config True if config is the starting configuration
                             (this matters only for directed graphs)
    """
    # new index of the configuration
    index = G.number_of_nodes()
    G.add_node(index, config=tuple(config))
    G_configs = nx.get_node_attributes(G, 'config')
    G_configs = [G_configs[node] for node in G_configs]
    
    # TODO: Implement here
    # Add edges from the newly added node
    # Support lazy=True
    edges=[]
    global evaluated
    for i in range(len(G_configs)):
	if(index==i):
		continue
	dist = np.linalg.norm(np.array(G_configs[i])-np.array(config),axis=0)
	if(dist==0):
		continue
	if(dist < connection_radius):
		if(not lazy):			
			evaluated = evaluated+1
			if(start_from_config):			
				valid, cost = env.edge_validity_checker(config,G_configs[i])
			else:
				valid, cost = env.edge_validity_checker(G_configs[i],config)						
		else: # Lazy case
			if(start_from_config):			
				cost = env.compute_heuristic(config,G_configs[i])
			else:
				cost = env.compute_heuristic(G_configs[i],config)                        			         
			valid=True					
		if(valid):
			if(cost!=0):
				if(start_from_config):
					edges.append((index,i,cost))
										
				else:
					edges.append((i,index,cost))					
        
									
    G.add_weighted_edges_from(edges)
    # Check for connectivity.
    num_connected_components = 1#len(list(nx.connected_components(G)))
    if not num_connected_components == 1:
        print ("warning, Graph has {} components, not connected".format(num_connected_components))

    return G, index

