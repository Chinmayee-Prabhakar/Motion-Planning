# Standard Algorithm Implementation
# Sampling-based Algorithms PRM

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from scipy.spatial import KDTree

# Class for PRM
class PRM:
    # Constructor
    def __init__(self, map_array):
        self.map_array = map_array            # map array, 1->free, 0->obstacle
        self.size_row = map_array.shape[0]    # map size
        self.size_col = map_array.shape[1]    # map size

        self.samples = []                     # list of sampled points
        self.graph = nx.Graph()               # constructed graph
        self.path = []                        # list of nodes of the found path
        #print(map_array)

    def check_collision(self, p1, p2):
        '''Check if the path between two points collide with obstacles
        arguments:
            p1 - point 1, [row, col]
            p2 - point 2, [row, col]

        return:
            True if there are obstacles between two points
        '''
        ### YOUR CODE HERE ###

        pts_row = np.linspace(p1[0], p2[0])
        pts_col = np.linspace(p1[1], p2[1])
        pts_inbtwn = zip(pts_row, pts_col)
        for p in pts_inbtwn:
            if (self.map_array[int(p[0])][int(p[1])] != 1):
                return True
        return False


    def dis(self, point1, point2):
        '''Calculate the euclidean distance between two points
        arguments:
            p1 - point 1, [row, col]
            p2 - point 2, [row, col]

        return:
            euclidean distance between two points
        '''
        ### YOUR CODE HERE ###
        H = (point1[0] - point2[0])**2 + (point1[1] - point2[1])**2
        len = np.sqrt(H)
        return len


    def uniform_sample(self, n_pts):
        '''Use uniform sampling and store valid points
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points

        check collision and append valid points to self.samples
        as [(row1, col1), (row2, col2), (row3, col3) ...]
        '''
        # Initialize graph
        self.graph.clear()

        ### YOUR CODE HERE ###
        rows = int(np.sqrt(n_pts * (self.size_row / self.size_col)))
        col = int(n_pts / rows)
        r = np.linspace(0, self.size_row - 1, num=rows, dtype=int)
        c = np.linspace(0, self.size_col - 1, num=col, dtype=int)
        pt_r, pt_c = np.meshgrid(r, c)
        sm_pts = list(zip(pt_r.flatten(), pt_c.flatten()))
        for p in sm_pts:
            if self.map_array[int(p[0])][int(p[1])] == 1:
                self.samples.append((int(p[0]), int(p[1])))
    
    def random_sample(self, n_pts):
        '''Use random sampling and store valid points
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points

        check collision and append valide points to self.samples
        as [(row1, col1), (row2, col2), (row3, col3) ...]
        '''
        # Initialize graph
        self.graph.clear()

        ### YOUR CODE HERE ###
        xy_min = [0, 0]
        xy_max = [self.size_row - 1, self.size_col - 1]
        dt = np.random.uniform(low=xy_min, high=xy_max, size=(n_pts, 2))
        for d in dt:
            if self.map_array[int(d[0])][int(d[1])] == 1:
                self.samples.append((int(d[0]), int(d[1])))


    def gaussian_sample(self, n_pts):
        '''Use gaussian sampling and store valid points
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points

        check collision and append valide points to self.samples
        as [(row1, col1), (row2, col2), (row3, col3) ...]
        '''
        # Initialize graph
        self.graph.clear()

        ### YOUR CODE HERE ###
        G_deviation = [[100, 0], [0, 100]]
        pt_2 = []
        xy_min = [0, 0]
        xy_max = [self.size_row - 1, self.size_col - 1]
        dt = np.random.uniform(low=xy_min, high=xy_max, size=(n_pts, 2))

        for p in dt:
            pt = np.random.multivariate_normal(p, G_deviation, size=1)
            for ps in pt:
                pt_2 = [int(ps[0]), int(ps[1])]
            #print("p:", pt_2)
            #print(p)
            if (self.size_row <= pt_2[0]) or (pt_2[0] < 0) or (self.size_col <= pt_2[1]) or (pt_2[1] < 0):
                continue
            point_1 = self.map_array[int(p[0])][int(p[1])]
            point_2 = self.map_array[int(pt_2[0])][int(pt_2[1])]
            if point_1 == 1 and point_2 == 0:
                self.samples.append((int(p[0]), int(p[1])))
            elif point_1 == 0 and point_2 == 1:
                self.samples.append((int(pt_2[0]), int(pt_2[1])))

    def bridge_sample(self, n_pts):
        '''Use bridge sampling and store valid points
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points

        check collision and append valide points to self.samples
        as [(row1, col1), (row2, col2), (row3, col3) ...]
        '''
        # Initialize graph
        self.graph.clear()

        ### YOUR CODE HERE ###
        B_deviation = [[400, 0], [0, 400]]
        pt_1 = []
        pt_2 = []

        for h in range(n_pts):
            if (self.map_array[np.random.randint(0, self.size_row)][np.random.randint(0, self.size_col)] == 0):
                pt_1.append([np.random.randint(0, self.size_row), np.random.randint(0, self.size_col)])
        for p in pt_1:
            pt = np.random.multivariate_normal(p, B_deviation, size=1)
            for ps in pt:
                pt_2 = [int(ps[0]), int(ps[1])]
            if (self.size_row <= pt_2[0]) or (pt_2[0] < 0) or (self.size_col <= pt_2[1]) or (pt_2[1] < 0):
                continue
            if (self.map_array[pt_2[0]][pt_2[1]] == 0):
                mdl = (round((p[0] + pt_2[0]) / 2), round((p[1] + pt_2[1]) / 2))
                if (self.map_array[mdl[0]][mdl[1]] == 1):
                    self.samples.append(mdl)


    def draw_map(self):
        '''Visualization of the result
        '''
        # Create empty map
        fig, ax = plt.subplots()
        img = 255 * np.dstack((self.map_array, self.map_array, self.map_array))
        ax.imshow(img)

        # Draw graph
        # get node position (swap coordinates)
        node_pos = np.array(self.samples)[:, [1, 0]]
        pos = dict( zip( range( len(self.samples) ), node_pos) )
        pos['start'] = (self.samples[-2][1], self.samples[-2][0])
        pos['goal'] = (self.samples[-1][1], self.samples[-1][0])
        
        # draw constructed graph
        nx.draw(self.graph, pos, node_size=3, node_color='y', edge_color='y' ,ax=ax)

        # If found a path
        if self.path:
            # add temporary start and goal edge to the path
            final_path_edge = list(zip(self.path[:-1], self.path[1:]))
            nx.draw_networkx_nodes(self.graph, pos=pos, nodelist=self.path, node_size=8, node_color='b')
            nx.draw_networkx_edges(self.graph, pos=pos, edgelist=final_path_edge, width=2, edge_color='b')

        # draw start and goal
        nx.draw_networkx_nodes(self.graph, pos=pos, nodelist=['start'], node_size=12,  node_color='g')
        nx.draw_networkx_nodes(self.graph, pos=pos, nodelist=['goal'], node_size=12,  node_color='r')

        # show image
        plt.axis('on')
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.show()


    def sample(self, n_pts=1000, sampling_method="uniform"):
        '''Construct a graph for PRM
        arguments:
            n_pts - number of points try to sample, 
                    not the number of final sampled points
            sampling_method - name of the chosen sampling method

        Sample points, connect, and add nodes and edges to self.graph
        '''
        # Initialize before sampling
        self.samples = []
        self.graph.clear()
        self.path = []

        # Sample methods
        if sampling_method == "uniform":
            self.uniform_sample(n_pts)
        elif sampling_method == "random":
            self.random_sample(n_pts)
        elif sampling_method == "gaussian":
            self.gaussian_sample(n_pts)
        elif sampling_method == "bridge":
            self.bridge_sample(n_pts)

        ### YOUR CODE HERE ###

        # Find the pairs of points that need to be connected
        # and compute their distance/weight.
        # Store them as
        # pairs = [(p_id0, p_id1, weight_01), (p_id0, p_id2, weight_02), 
        #          (p_id1, p_id2, weight_12) ...]
        pairs = []
        Nearest_neighbors = 18
        kdt = KDTree(self.samples)

        pr = kdt.query_pairs(r=Nearest_neighbors)

        for p_ in pr:
            p_one = self.samples[p_[0]]
            p_two = self.samples[p_[1]]
            weights = self.dis(p_one, p_two)
            if self.check_collision(p_one, p_two):
                continue
            else:
                pairs.append((p_[0], p_[1], weights))

        # Use sampled points and pairs of points to build a graph.
        # To add nodes to the graph, use
        # self.graph.add_nodes_from([p_id0, p_id1, p_id2 ...])
        # To add weighted edges to the graph, use
        # self.graph.add_weighted_edges_from([(p_id0, p_id1, weight_01), 
        #                                     (p_id0, p_id2, weight_02), 
        #                                     (p_id1, p_id2, weight_12) ...])
        # 'p_id' here is an integer, representing the order of 
        # current point in self.samples
        # For example, for self.samples = [(1, 2), (3, 4), (5, 6)],
        # p_id for (1, 2) is 0 and p_id for (3, 4) is 1.
        self.graph.add_nodes_from([])
        self.graph.add_weighted_edges_from(pairs)

        # Print constructed graph information
        n_nodes = self.graph.number_of_nodes()
        n_edges = self.graph.number_of_edges()
        print("The constructed graph has %d nodes and %d edges" %(n_nodes, n_edges))


    def search(self, start, goal):
        '''Search for a path in graph given start and goal location
        arguments:
            start - start point coordinate [row, col]
            goal - goal point coordinate [row, col]

        Temporary add start and goal node, edges of them and their nearest neighbors
        to graph for self.graph to search for a path.
        '''
        # Clear previous path
        self.path = []

        # Temporarily add start and goal to the graph
        self.samples.append(start)
        self.samples.append(goal)
        # start and goal id will be 'start' and 'goal' instead of some integer
        self.graph.add_nodes_from(['start', 'goal'])

        ### YOUR CODE HERE ###

        # Find the pairs of points that need to be connected
        # and compute their distance/weight.
        # You could store them as
        # start_pairs = [(start_id, p_id0, weight_s0), (start_id, p_id1, weight_s1), 
        #                (start_id, p_id2, weight_s2) ...]
        start_pairs = []
        goal_pairs = []
        Neighbor_sample = 25
        kdt = KDTree(self.samples)
        pts, ps = kdt.query([start, goal], Neighbor_sample)
        #print(Neighbor_sample)
        #print('ps:', ps)
        for j in range(Neighbor_sample):
            weights = self.dis(self.samples[ps[0][j]], start)
            if not(self.check_collision(self.samples[ps[0][j]], start)):
                start_pairs.append(('start', ps[0][j], weights))
            weights = self.dis(self.samples[ps[1][j]], goal)
            if not(self.check_collision(self.samples[ps[1][j]], goal)):
                goal_pairs.append(('goal', ps[1][j], weights))


        # Add the edge to graph
        self.graph.add_weighted_edges_from(start_pairs)
        self.graph.add_weighted_edges_from(goal_pairs)
        
        # Seach using Dijkstra
        try:
            self.path = nx.algorithms.shortest_paths.weighted.dijkstra_path(self.graph, 'start', 'goal')
            path_length = nx.algorithms.shortest_paths.weighted.dijkstra_path_length(self.graph, 'start', 'goal')
            print("The path length is %.2f" %path_length)
        except nx.exception.NetworkXNoPath:
            print("No path found")
        
        # Draw result
        self.draw_map()

        # Remove start and goal node and their edges
        self.samples.pop(-1)
        self.samples.pop(-1)
        self.graph.remove_nodes_from(['start', 'goal'])
        self.graph.remove_edges_from(start_pairs)
        self.graph.remove_edges_from(goal_pairs)
        