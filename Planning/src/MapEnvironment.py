import numpy as np
from matplotlib import pyplot as plt

class MapEnvironment(object):

    def __init__(self, map_data, stepsize=0.05):
        """
        @param map_data: 2D numpy array of map
        @param stepsize: size of a step to generate waypoints
        """
        # Obtain the boundary limits.
        # Check if file exists.
        self.map = map_data
        self.xlimit = [0, np.shape(self.map)[0]]
        self.ylimit = [0, np.shape(self.map)[1]]
        self.limit = np.array([self.xlimit, self.ylimit])
        self.maxdist = np.float('inf')
        self.stepsize = stepsize

        # Display the map. Uncomment
        '''plt.imshow(self.map, interpolation='nearest', origin='lower')
        plt.savefig('map.png')
        print("Saved map as map.png")'''

    def state_validity_checker(self, configs):
        """
        @param configs: 2D array of [num_configs, dimension].
        Each row contains a configuration
        @return numpy list of boolean values. True for valid states.
        """
	configs = np.array(configs)
        if len(configs.shape) == 1:
            configs = configs.reshape(1, -1)

        # Implement here
        # 1. Check for state bounds within xlimit and ylimit
	# 2. Check collision
	
	validity = np.zeros(configs.shape[0],dtype=np.int)
	for i in range(configs.shape[0]):
		if((configs[i,0]<self.xlimit[1]) and (configs[i,1]<self.ylimit[1]) and (configs[i,0]>=self.xlimit[0]) and (configs[i,1]>=self.ylimit[0])):
			if(self.map[int(configs[i,0]),int(configs[i,1])]==0):
				validity[i] = 1
	
        return validity

    def edge_validity_checker(self, config1, config2):
        """
        Checks whether the path between config 1 and config 2 is valid
        """
	path, length = self.generate_path(config1, config2)
	if length == 0:
            return False, 0
	valid = self.state_validity_checker(path)		
	if not np.all(valid):
            return False, self.maxdist
	return True, length

    def compute_heuristic(self, config, goal):
        """
        Returns a heuristic distance between config and goal
        @return a float value
        """
        # Implement here
	heuristic = ((config[0]-goal[0])**2 + (config[1]-goal[1])**2)**0.5

        return heuristic

    def compute_distances(self, start_config, end_configs):
        """
        Compute distance from start_config and end_configs in L2 metric.
        This function performs what compute_heuristic does, but across
        multiple configurations.
        @param start_config: tuple of start config
        @param end_configs: list of tuples of end confings
        @return 1D  numpy array of distances
        """
	dist = np.zeros(np.shape(end_configs)[0])
	count=0
        for config in end_configs:
		dist[count] = np.linalg.norm(np.array(start_config)-np.array(config))
		count+=1
		
        return dist

    def generate_path(self, config1, config2):
        config1 = np.array((config1))
        config2 = np.array((config2))
	dist = np.linalg.norm(config2 - config1)
        if dist == 0:
            return config1, dist
        direction = (config2 - config1) / dist
        steps = dist // self.stepsize + 1	

        waypoints = np.array([np.linspace(config1[i], config2[i], steps) for i in range(2)]).transpose()
	
        return waypoints, dist

    def get_path_on_graph(self, G, path_nodes):
        plan = []
        for node in path_nodes:
            plan += [G.nodes[node]["config"]]
        plan = np.array(plan)

        path = []
        xs, ys, yaws = [], [], []
        for i in range(np.shape(plan)[0] - 1):
            path += [self.generate_path(plan[i], plan[i+1])[0]]

        return np.concatenate(path, axis=0)

    def shortcut(self, G, waypoints, num_trials=100):
        """
        Short cut waypoints if collision free
        @param waypoints list of node indices in the graph
        """
        print("Originally {} waypoints".format(len(waypoints)))
	edges=[]
        for _ in range(num_trials):

            if len(waypoints) == 2:
                break
	    
            # Implement here
	    
            # 1. Choose two configurations
	    nodes = np.random.choice(waypoints, size=2, replace=False)
	    i1 = np.where(waypoints==nodes[0])[0][0]
	    i2 = np.where(waypoints==nodes[1])[0][0]
	    
	    while(abs(i1-i2)==1):
		nodes = np.random.choice(waypoints, size=2, replace=False)
	    	i1 = np.where(waypoints==nodes[0])[0][0]
	    	i2 = np.where(waypoints==nodes[1])[0][0]	    
	    # 2. Check for collision
	    minnode = min(i1,i2)
	    maxnode = max(i1,i2)
	    inp1 = G.nodes[waypoints[minnode]]['config']
	    inp2 = G.nodes[waypoints[maxnode]]['config']
	    valid, cost = self.edge_validity_checker(inp1,inp2)
	    # 3. Connect them if collision free
	    if(valid):
		edges.append((waypoints[minnode],waypoints[maxnode],cost))
		waypoints = waypoints[:minnode+1]+waypoints[maxnode:]	
	
	G.add_weighted_edges_from(edges)
        print("Path shortcut to {} waypoints".format(len(waypoints)))
        return waypoints

    def visualize_plan(self, G, path_nodes, hide_graph=False, saveto="plan.png", show=True):
        '''
        Visualize the final path
        We flip x,y and subtract by 0.5 just for visualization purpose
        (to aligh with imshow's map visualization).
        @param plan Sequence of states defining the plan.
        '''
        plan = []
        for node in path_nodes:
            plan += [G.nodes[node]["config"]]
        plan = np.array(plan)

        plt.clf()
        plt.imshow(self.map, interpolation='none', cmap='gray', origin='lower')

        # Comment this to hide all edges. This can take long.
        if not hide_graph:
            edges = G.edges()
            for edge in edges:
                config1 = G.nodes[edge[0]]["config"]
                config2 = G.nodes[edge[1]]["config"]
                path, _ = self.generate_path(config1, config2)
                path = np.array(path).astype(np.float64) - 0.5
                plt.plot(path[:,1], path[:, 0], 'grey')

        path = self.get_path_on_graph(G, path_nodes) - 0.5
        plt.plot(path[:,1], path[:,0], 'y', linewidth=5)

        for vertex in G.nodes:
            config = np.array(G.nodes[vertex]["config"]) - 0.5
            plt.scatter(config[1], config[0], s=10, c='r')

        plt.tight_layout()

        plt.ylabel('x')
        plt.xlabel('y')
        plt.xlim(self.ylimit[0]-0.5, self.ylimit[1]-0.5)
        plt.ylim(self.xlimit[0]-0.5, self.xlimit[1]-0.5)
        plt.axis('off')

	#uncomment
        if saveto != "":
            plt.savefig(saveto)
            print("Saved the plan to {}".format(saveto))

        if show:
            plt.show()

    def visualize_graph(self, G, saveto="graph.png"):
        # We flip x,y and subtract by 0.5 just for visualization purpose
        # (to aligh with imshow's map visualization).

        plt.clf()
        plt.imshow(self.map, interpolation='nearest', origin='lower')
        edges = G.edges()
        for edge in edges:
            config1 = G.nodes[edge[0]]["config"]
            config2 = G.nodes[edge[1]]["config"]
            path = self.generate_path(config1, config2)[0].astype(np.float64) - 0.5
            plt.plot(path[:,1], path[:,0], 'w')
            # Uncomment this to see all waypoints.
            #plt.scatter(path[:,1], path[:,0], c='w', marker='+')

        num_nodes = G.number_of_nodes()
        for i, vertex in enumerate(G.nodes):
            config = np.array(G.nodes[vertex]["config"]).astype(np.float64) - 0.5
            if i == num_nodes - 2:
                # Color the start node with blue
                plt.scatter(config[1], config[0], s=30, c='b')
            elif i == num_nodes - 1:
                # Color the goal node with green
                plt.scatter(config[1], config[0], s=30, c='g')
            else:
                plt.scatter(config[1], config[0], s=30, c='r')

        plt.tight_layout()
        plt.xlim(self.ylimit[0]-0.5, self.ylimit[1]-0.5)
        plt.ylim(self.xlimit[0]-0.5, self.xlimit[1]-0.5)
        plt.axis('off')

        plt.ylabel('x')
        plt.xlabel('y')

        if saveto != "":
            plt.savefig(saveto)
            print("Saved the graph to {}".format(saveto))

        plt.show()
