#!/usr/bin/env python

import argparse, numpy, time
import networkx as nx
import numpy as np

from MapEnvironment import MapEnvironment
import graph_maker
import astar
import lazy_astar
from Sampler import Sampler


# Try running the following
# python run.py -m ../maps/map1.txt -s 1 1 -g 8 7 --num-vertices 15
# python run.py -m ../maps/map2.txt -s 321 148 -g 106 202 --num-vertices 300 --connection-radius 100

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='script for testing planners')

    parser.add_argument('-m', '--map', type=str, default='map1.txt',
                        help='The environment to plan on')
    parser.add_argument('-s', '--start', nargs='+', type=int, required=True)
    parser.add_argument('-g', '--goal', nargs='+', type=int, required=True)
    parser.add_argument('--num-vertices', type=int, required=True)
    parser.add_argument('--connection-radius', type=float, default=5.0)
    parser.add_argument('--lazy', action='store_true')
    parser.add_argument('--shortcut', action='store_true')

    args = parser.parse_args()

    # First setup the environment
    map_data = np.loadtxt(args.map).astype(np.int)
    planning_env = MapEnvironment(map_data)

    # Make a graph
    start_time = time.time()
    G = graph_maker.make_graph(planning_env,
        sampler=Sampler(planning_env),
        num_vertices=args.num_vertices,
        connection_radius=args.connection_radius,
        lazy=args.lazy,
        directed=False)
    #uncomment
    print("Graph creation time: %s seconds" % (time.time() - start_time))

    # Add start and goal nodes
    G, start_id = graph_maker.add_node(G, args.start, env=planning_env,
        connection_radius=args.connection_radius, start_from_config=True, lazy=args.lazy)
    G, goal_id = graph_maker.add_node(G, args.goal, env=planning_env,
        connection_radius=args.connection_radius, start_from_config=False, lazy=args.lazy)

    # Uncomment this to visualize the graph. (This takes time for a large graph)
    planning_env.visualize_graph(G)

    # heuristic weight
    a = 1.0

    try:
        heuristic = lambda n1, n2: planning_env.compute_heuristic(
            G.nodes[n1]['config'], G.nodes[n2]['config'])

        if args.lazy:
            weight = lambda n1, n2: planning_env.edge_validity_checker(
                G.nodes[n1]['config'], G.nodes[n2]['config'])
	    start_time = time.time()
            path = lazy_astar.astar_path(G,
                source=start_id, target=goal_id, weight=weight, heuristic=heuristic, a=a)
	    print("Lazy A* planning time: %s seconds" % (time.time() - start_time))
	    astar_time = time.time()-start_time
	    print("Path Length: " + str(astar.path_length(G, path)))
        else:
	    print("")
	    start_time = time.time()
            path = astar.astar_path(G,
                source=start_id, target=goal_id, heuristic=heuristic, a=a)
	    astar_time = time.time()-start_time
	    print("A* planning time: ", astar_time)
	    print("Path Length: " + str(astar.path_length(G, path)))

        planning_env.visualize_plan(G, path)

        if args.shortcut:
	    start_time = time.time()
            shortcut_path = planning_env.shortcut(G, path)
	    shortcut_time = time.time()-start_time
	    print("")
	    print("Shortcut planning time: ",shortcut_time)
	    print("Total planning time: ",shortcut_time+astar_time)
	    print("Shortcut Path Length: " + str(astar.path_length(G, shortcut_path)))
	    planning_env.visualize_plan(G, shortcut_path, 'shortcut_plan.png')

    except nx.NetworkXNoPath as e:
        print(e)