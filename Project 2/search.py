# Basic searching algorithms

# Class for each node in the grid
class Node:
    def __init__(self, row, col, is_obs, h):
        self.row = row        # coordinate
        self.col = col        # coordinate
        self.is_obs = is_obs  # obstacle?
        self.g = None         # cost to come (previous g + moving cost)
        self.h = h            # heuristic
        self.cost = None      # total cost (depend on the algorithm)
        self.parent = None    # previous node

def dijkstra(grid, start, goal):
    '''Return a path found by Dijkstra alogirhm 
       and the number of steps it takes to find it.

    arguments:
    grid - A nested list with datatype int. 0 represents free space while 1 is obstacle.
           e.g. a 3x3 2D map: [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    start - The start node in the map. e.g. [0, 0]
    goal -  The goal node in the map. e.g. [2, 2]

    return:
    path -  A nested list that represents coordinates of each step (including start and goal node), 
            with data type int. e.g. [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2]]
    steps - Number of steps it takes to find the final solution, 
            i.e. the number of nodes visited before finding a path (including start and goal node)

    >>> from main import load_map
    >>> grid, start, goal = load_map('test_map.csv')
    >>> dij_path, dij_steps = dijkstra(grid, start, goal)
    It takes 10 steps to find a path using Dijkstra
    >>> dij_path
    [[0, 0], [1, 0], [2, 0], [3, 0], [3, 1]]
    '''
    ### YOUR CODE HERE ###
    path = []
    steps = 0
    found = False

    Source = Node(start[0], start[1], False, 1)  # Start Node
    Goal = Node(goal[0], goal[1], False, 1)

    Adj = []

    row = []  # List of row co-ordinates
    col = []  # List of col co-ordinates
    Obstacle = []  # List of Obstacle nodes
    obst_free = []


    Parent = []
    Cost = {}
    Cost[(Source.row, Source.col)] = 0
    Costs = {}
    Costs[(Source.row, Source.col)] = 0
    Value = float('inf')
    Complete = []


    grid_rows = len(grid)
    grid_col = len(grid[0])

    for i in range(grid_rows):
        j = 0
        rw = grid[i]
        for j in range(grid_col):
            if rw[j] == 1:
                Obstacle.append([i, j])
                Cost[(i, j)] = Value
                j += 1

            else:
                obst_free.append([i, j])
                Cost[(i, j)] = Value
                j += 1
        i += 1

    Cost[(Source.row, Source.col)] = 0
    Costs[(Source.row, Source.col)] = 0

    while len(Cost) >= 1:

           k = min(Cost, key=Cost.get)
           node = [k[0], k[1]]

           if node in Obstacle:
               p = Cost.pop(k, None)
               q = Costs.pop(k, None)
               continue
           elif node in obst_free:
               if node == goal:
                   found = True
                   steps += 1
                   break
               elif node == [Source.row, Source.col]:

                   N = Node(node[0], node[1], False, 1)
                   if N.row == 0:
                       Exp_it_row = [1]
                   elif N.row == 9:
                       Exp_it_row = [-1]
                   else:
                       Exp_it_row = [1, -1]

                   if N.col == 0:
                       Exp_it_col = [1]
                   elif N.col == 9:
                       Exp_it_col = [-1]
                   else:
                       Exp_it_col = [1, -1]

                   i = 0
                   if i < len(Exp_it_row):
                       for increment in Exp_it_row:
                           row.append(N.row + increment)
                           i += 1
                   i = 0
                   if i < len(Exp_it_col):
                       for increment in Exp_it_col:
                           col.append(N.col + increment)
                           i += 1

                   for c in range(len(col)):
                       Adj.append([N.row, col[c]])
                       for r in range(len(row)):
                           Adj.append([row[r], N.col])

                   N.parent = None
                   Parent.append(N.parent)
                   steps += 1
                   Complete.append(node)

               else:
                   Adj.clear()
                   N = Node(node[0], node[1], False, 1)
                   if N.row == 0:
                       Exp_it_row = [1]
                   elif N.row == 9:
                       Exp_it_row = [-1]
                   else:
                       Exp_it_row = [1, -1]

                   if N.col == 0:
                       Exp_it_col = [1]
                   elif N.col == 9:
                       Exp_it_col = [-1]
                   else:
                       Exp_it_col = [1, -1]

                   i = 0
                   row.clear()
                   col.clear()
                   if i < len(Exp_it_row):
                       for increment in Exp_it_row:
                           row.append(N.row + increment)
                           i += 1
                   i = 0
                   if i < len(Exp_it_col):
                       for increment in Exp_it_col:
                           col.append(N.col + increment)
                           i += 1

                   for c in range(len(col)):
                       Adj.append([N.row, col[c]])
                       for r in range(len(row)):
                           Adj.append([row[r], N.col])
                   steps += 1
                   Complete.append(node)

               for v in Adj:
                 if v not in Complete:
                   V = Node(v[0], v[1], False, 1)
                   V.parent = node
                   Parent.append([v, V.parent])

                   if (V.row, V.col) in Cost:
                       if Cost[(V.row, V.col)] > Cost[(node[0], node[1])] + 1:
                           Cost[(V.row, V.col)] = Cost[(node[0], node[1])] + 1
                           Costs[(V.row, V.col)] = Costs[(node[0], node[1])] + 1

               p = Cost.pop(k, None)


    if found:
        w = Parent.pop(0)
        Nde = goal
        path.append(goal)
        while Nde != start:
            for nd in Parent:

                if nd[0] == Nde:
                    Nde = nd[1]
                    path.insert(0, Nde)




    if found:
        print(f"It takes {steps} steps to find a path using Dijkstra")
    else:
        print("No path found")
    return path, steps


def astar(grid, start, goal):
    '''Return a path found by A* alogirhm 
       and the number of steps it takes to find it.

    arguments:
    grid - A nested list with datatype int. 0 represents free space while 1 is obstacle.
           e.g. a 3x3 2D map: [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    start - The start node in the map. e.g. [0, 0]
    goal -  The goal node in the map. e.g. [2, 2]

    return:
    path -  A nested list that represents coordinates of each step (including start and goal node), 
            with data type int. e.g. [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2]]
    steps - Number of steps it takes to find the final solution, 
            i.e. the number of nodes visited before finding a path (including start and goal node)

    >>> from main import load_map
    >>> grid, start, goal = load_map('test_map.csv')
    >>> astar_path, astar_steps = astar(grid, start, goal)
    It takes 7 steps to find a path using A*
    >>> astar_path
    [[0, 0], [1, 0], [2, 0], [3, 0], [3, 1]]
    '''
    ### YOUR CODE HERE ###
    path = []
    steps = 0
    found = False

    Source = Node(start[0], start[1], False, 1)  # Start Node
    Goal = Node(goal[0], goal[1], False, 1)

    Adj = []

    row = []  # List of row co-ordinates
    col = []  # List of col co-ordinates
    Obstacle = []  # List of Obstacle nodes
    obst_free = []
    Q = []
    Q.append([Source.row, Source.col])

    Parent = []
    Cost = {}
    Cost[(Source.row, Source.col)] = 0
    Costs = {}
    Costs[(Source.row, Source.col)] = 0
    Value = float('inf')
    Complete = []
    H = {}
    H[(Source.row, Source.col)] = abs(Source.row - Goal.row) + abs(Source.col - Goal.col)
    Q = {}
    Q[(Source.row, Source.col)] = 0

    grid_rows = len(grid)
    grid_col = len(grid[0])

    for i in range(grid_rows):
        j = 0
        rw = grid[i]
        for j in range(grid_col):
            if rw[j] == 1:
                Obstacle.append([i, j])
                Cost[(i, j)] = Value
                H[(i, j)] = abs(i - Goal.row) + abs(j - Goal.col)
                j += 1

            else:
                obst_free.append([i, j])
                Cost[(i, j)] = Value
                H[(i, j)] = abs(i - Goal.row) + abs(j - Goal.col)
                j += 1
        i += 1

    Cost[(Source.row, Source.col)] = 0
    Costs[(Source.row, Source.col)] = 0
    #print(H)

    while True:
           k = min(Q, key=Q.get)
           node = [k[0], k[1]]

           if node in Obstacle:
               p = Cost.pop(k, None)
               q = Costs.pop(k, None)
               r = Q.pop(k, None)
               continue
           elif node in obst_free:
               if node == goal:
                   found = True
                   steps += 1
                   break
               elif node == [Source.row, Source.col]:

                   N = Node(node[0], node[1], False, 1)
                   if N.row == 0:
                       Exp_it_row = [1]
                   elif N.row == 9:
                       Exp_it_row = [-1]
                   else:
                       Exp_it_row = [1, -1]

                   if N.col == 0:
                       Exp_it_col = [1]
                   elif N.col == 9:
                       Exp_it_col = [-1]
                   else:
                       Exp_it_col = [1, -1]

                   i = 0
                   if i < len(Exp_it_row):
                       for increment in Exp_it_row:
                           row.append(N.row + increment)
                           i += 1
                   i = 0
                   if i < len(Exp_it_col):
                       for increment in Exp_it_col:
                           col.append(N.col + increment)
                           i += 1

                   for c in range(len(col)):
                       Adj.append([N.row, col[c]])
                       for r in range(len(row)):
                           Adj.append([row[r], N.col])

                   N.parent = None
                   Parent.append(N.parent)
                   steps += 1
                   Complete.append(node)

               else:
                   Adj.clear()
                   N = Node(node[0], node[1], False, 1)
                   if N.row == 0:
                       Exp_it_row = [1]
                   elif N.row == 9:
                       Exp_it_row = [-1]
                   else:
                       Exp_it_row = [1, -1]

                   if N.col == 0:
                       Exp_it_col = [1]
                   elif N.col == 9:
                       Exp_it_col = [-1]
                   else:
                       Exp_it_col = [1, -1]

                   i = 0
                   row.clear()
                   col.clear()
                   if i < len(Exp_it_row):
                       for increment in Exp_it_row:
                           row.append(N.row + increment)
                           i += 1
                   i = 0
                   if i < len(Exp_it_col):
                       for increment in Exp_it_col:
                           col.append(N.col + increment)
                           i += 1

                   for c in range(len(col)):
                       Adj.append([N.row, col[c]])
                       for r in range(len(row)):
                           Adj.append([row[r], N.col])
                   steps += 1
                   Complete.append(node)

               for v in Adj:
                 if v not in Complete:
                   V = Node(v[0], v[1], False, 1)
                   V.parent = node
                   Parent.append([v, V.parent])
                   if (V.row, V.col) in Cost:
                       if Cost[(V.row, V.col)] > Cost[(node[0], node[1])] + 1:
                           Cost[(V.row, V.col)] = Cost[(node[0], node[1])] + 1
                           Costs[(V.row, V.col)] = Costs[(node[0], node[1])] + 1
                           Q[(V.row, V.col)] = H[(V.row, V.col)] + Cost[(V.row, V.col)]

               p = Cost.pop(k, None)
               q = Q.pop(k, None)


    if found:
        w = Parent.pop(0)
        Nde = goal
        path.append(goal)
        while Nde != start:
            for nd in Parent:

                if nd[0] == Nde:
                    Nde = nd[1]
                    path.insert(0, Nde)


    if found:
        print(f"It takes {steps} steps to find a path using A*")
    else:
        print("No path found")
    return path, steps


# Doctest
if __name__ == "__main__":
    # load doc test
    from doctest import testmod, run_docstring_examples
    # Test all the functions
    testmod()
