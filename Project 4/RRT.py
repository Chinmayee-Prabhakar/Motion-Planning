# Standard Algorithm Implementation
# Sampling-based Algorithms RRT and RRT*

import matplotlib.pyplot as plt
import numpy as np
import math

# Class for each tree node
class Node:
    def __init__(self, row, col):
        self.row = row        # coordinate
        self.col = col        # coordinate
        self.parent = None    # parent node
        self.cost = 0.0       # cost


# Class for RRT
class RRT:
    # Constructor
    def __init__(self, map_array, start, goal):
        self.map_array = map_array            # map array, 1->free, 0->obstacle
        self.size_row = map_array.shape[0]    # map size
        self.size_col = map_array.shape[1]    # map size

        self.start = Node(start[0], start[1]) # start node
        self.goal = Node(goal[0], goal[1])    # goal node
        self.vertices = []                    # list of nodes
        self.found = False                    # found flag


    def init_map(self):
        '''Intialize the map before each search
        '''
        self.found = False
        self.vertices = []
        self.vertices.append(self.start)


    def dis(self, node1, node2):
        '''Calculate the euclidean distance between two nodes
        arguments:
            node1 - node 1
            node2 - node 2

        return:
            euclidean distance between two nodes
        '''
        ### YOUR CODE HERE ###
        H = (node1.row - node2.row) ** 2 + (node1.col - node2.col) ** 2
        len = np.sqrt(H)
        return len



    def check_collision(self, node1, node2):
        '''Check if the path between two nodes collide with obstacles
        arguments:
            node1 - node 1
            node2 - node 2

        return:
            True if the new node is valid to be connected
        '''
        ### YOUR CODE HERE ###
        pts_row = np.linspace(node1.row, node2.row, num=50, dtype=int)
        pts_col = np.linspace(node1.col, node2.col, num=50, dtype=int)
        pts_inbtwn = zip(pts_row, pts_col)
        for p in pts_inbtwn:
            if (self.map_array[int(p[0])][int(p[1])] != 1):
                return False
        return True

    def get_new_point(self, goal_bias):
        '''Choose the goal or generate a random point
        arguments:
            goal_bias - the possibility of choosing the goal instead of a random point

        return:
            point - the new point
        '''
        ### YOUR CODE HERE ###
        rand_int = np.random.random()
        if rand_int < goal_bias:
            return self.goal
        else:
            flag = True
            while flag:
                row = np.random.randint(0, self.size_row-1)
                col = np.random.randint(0, self.size_col-1)
                if (self.map_array[row][col] == 0):
                    continue
                else:
                    point = Node(row, col)
                    flag = False
            return point


    def get_nearest_node(self, point):
        '''Find the nearest node in self.vertices with respect to the new point
        arguments:
            point - the new point

        return:
            the nearest node
        '''
        ### YOUR CODE HERE ###
        dist = math.inf
        for v in self.vertices:
            distance = self.dis(point, v)
            if distance < dist:
                dist = distance
                nearest_nde = v
        return nearest_nde


    def get_neighbors(self, new_node, neighbor_size):
        '''Get the neighbors that are within the neighbor distance from the node
        arguments:
            new_node - a new node
            neighbor_size - the neighbor distance

        return:
            neighbors - a list of neighbors that are within the neighbor distance
        '''
        ### YOUR CODE HERE ###
        neighbors = []
        vertices = self.vertices
        for v in vertices:
            vrow = v.row
            vcol = v.col
            if (self.map_array[vrow][vcol] != 0 ):
                d = self.dis(new_node, v)
                if d <= neighbor_size:
                    neighbors.append(v)
        return neighbors


    def rewire(self, new_node, neighbors):
        '''Rewire the new node and all its neighbors
        arguments:
            new_node - the new node
            neighbors - a list of neighbors that are within the neighbor distance from the node

        Rewire the new node if connecting to a new neighbor node will give least cost.
        Rewire all the other neighbor nodes.
        '''
        ### YOUR CODE HERE ###

        for N in neighbors:
            if N == new_node.parent:
                continue
            elif (N.cost > (new_node.cost + self.dis(new_node, N))) and self.check_collision(new_node, N):
                N.cost = new_node.cost + self.dis(new_node, N)
                N.parent = new_node

    def draw_map(self):
        '''Visualization of the result
        '''
        # Create empty map
        fig, ax = plt.subplots(1)
        img = 255 * np.dstack((self.map_array, self.map_array, self.map_array))
        ax.imshow(img)

        # Draw Trees or Sample points
        for node in self.vertices[1:-1]:
            plt.plot(node.col, node.row, markersize=3, marker='o', color='y')
            plt.plot([node.col, node.parent.col], [node.row, node.parent.row], color='y')

        # Draw Final Path if found
        if self.found:
            cur = self.goal
            while cur.col != self.start.col or cur.row != self.start.row:
                plt.plot([cur.col, cur.parent.col], [cur.row, cur.parent.row], color='b')
                cur = cur.parent
                plt.plot(cur.col, cur.row, markersize=3, marker='o', color='b')

        # Draw start and goal
        plt.plot(self.start.col, self.start.row, markersize=5, marker='o', color='g')
        plt.plot(self.goal.col, self.goal.row, markersize=5, marker='o', color='r')

        # show image
        plt.show()


    def RRT(self, n_pts=1000, search_area = 10):
        '''RRT main search function
        arguments:
            n_pts - number of points try to sample,
                    not the number of final sampled points

        In each step, extend a new node if possible, and check if reached the goal
        '''
        # Remove previous result
        self.init_map()

        ### YOUR CODE HERE ###
        min_dist_to_goal = 15
        for i in range(n_pts):

            point_nw = self.get_new_point(goal_bias = 0.05)
            closest_node = self.get_nearest_node(point_nw)
            btw_node_distance = self.dis(closest_node, point_nw)
            if btw_node_distance == 0:
                continue

            if (btw_node_distance <= search_area) and (point_nw != self.goal) and \
                    (self.map_array[closest_node.row][closest_node.col] == 0) and self.check_collision(point_nw, closest_node):
                new_node = point_nw

            else:

                rw = (point_nw.row - closest_node.row) * search_area / btw_node_distance
                cl = (point_nw.col - closest_node.col) * search_area / btw_node_distance
                nw_pt = [rw, cl]
                r = int(closest_node.row + nw_pt[0])
                c = int(closest_node.col + nw_pt[1])
                row_size = self.size_row - 1
                col_size = self.size_col - 1
                if (r < 0):
                    r = 0
                elif (r > (row_size + 1)):
                    r = row_size
                if (c < 0):
                    c = 0
                elif (c > (col_size + 1)):
                    c = col_size

                new_node = Node(r, c)
            if (self.map_array[new_node.row][new_node.col] == 1):
              if (self.check_collision(closest_node, new_node)):
                new_node.parent = closest_node
                new_node.cost = closest_node.cost + self.dis(new_node, closest_node)
                self.vertices.append(new_node)

            if (self.map_array[new_node.row][new_node.col] == 1) and ((self.dis(new_node, self.goal) <= min_dist_to_goal) and
                                                                      self.check_collision(new_node, self.goal)):
                self.found = True
                self.goal.parent = new_node
                self.goal.cost = new_node.cost + self.dis(new_node, self.goal)
                break

        if self.found:
            steps = len(self.vertices) - 2
            length = self.goal.cost
            print("It took %d nodes to find the current path" %steps)
            print("The path length is %.2f" %length)
        else:
            print("No path found")

        # Draw result
        self.draw_map()


    def RRT_star(self, n_pts=1000):
        '''RRT* search function
        arguments:
            n_pts - number of points try to sample,
                    not the number of final sampled points
            neighbor_size - the neighbor distance

        In each step, extend a new node if possible, and rewire the node and its neighbors
        '''
        # Remove previous result
        self.init_map()

        ### YOUR CODE HERE ###
        global neighbor_size
        neighbor_size = 20
        global search_area
        search_area = 10
        for i in range(n_pts):
            point_nw = self.get_new_point(goal_bias = 0.05)
            closest_node = self.get_nearest_node(point_nw)
            btw_node_distance = self.dis(closest_node, point_nw)
            if btw_node_distance == 0:
                continue
            if (btw_node_distance <= search_area) and (point_nw != self.goal) and (self.map_array[closest_node.row][closest_node.col] == 0) and self.check_collision(point_nw, closest_node):
                new_node = point_nw

            else:
                    rw = (point_nw.row - closest_node.row) * search_area / btw_node_distance
                    cl = (point_nw.col - closest_node.col) * search_area / btw_node_distance
                    nw_pt = [(point_nw.row - closest_node.row) * search_area / btw_node_distance, (point_nw.col - closest_node.col) * search_area / btw_node_distance]
                    r = int(closest_node.row + nw_pt[0])
                    c = int(closest_node.col + nw_pt[1])
                    row_size = self.size_row - 1
                    col_size = self.size_col - 1
                    if (r < 0):
                        r = 0
                    elif (r > (row_size)):
                        r = row_size
                    if (c < 0):
                        c = 0
                    elif (c > (col_size)):
                        c = col_size

                    new_node = Node(r, c)
            if (self.map_array[new_node.row][new_node.col] == 1) and (self.check_collision(closest_node, new_node)):
                neighbors = self.get_neighbors(new_node, neighbor_size)
                near_node = closest_node
                least_cost = closest_node.cost + self.dis(closest_node, new_node)
                for neighbor in neighbors:
                    if (self.check_collision(neighbor, new_node) and (self.map_array[neighbor.row][neighbor.col] == 1) and (neighbor.cost + self.dis(neighbor, new_node)) < least_cost):
                        near_node = neighbor
                        least_cost = neighbor.cost + self.dis(neighbor, new_node)
                new_node.parent = near_node
                new_node.cost = least_cost
                self.vertices.append(new_node)
                self.rewire(new_node, neighbors)

            for nbr in self.get_neighbors(self.goal, neighbor_size):
                if self.check_collision(nbr, self.goal):
                    if  (nbr.cost + self.dis(nbr, self.goal)) < self.goal.cost:
                        self.goal.parent = nbr
                        dist_node_goal = self.dis(nbr, self.goal)
                        self.goal.cost = nbr.cost + dist_node_goal
                        self.found = True

        # Output
        if self.found:
            steps = len(self.vertices) - 2
            length = self.goal.cost
            print("It took %d nodes to find the current path" %steps)
            print("The path length is %.2f" %length)
        else:
            print("No path found")

        # Draw result
        self.draw_map()
