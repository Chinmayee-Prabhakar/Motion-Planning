# Basic searching algorithms

# Class for each node in the grid
class Node:
    def __init__(self, row, col, is_obs):
        self.row = row        # coordinate
        self.col = col        # coordinate
        self.is_obs = is_obs  # obstacle?
        self.cost = None      # total cost (depend on the algorithm)
        self.parent = None    # previous node


def bfs(grid, start, goal):
    '''Return a path found by BFS alogirhm 
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
    >>> bfs_path, bfs_steps = bfs(grid, start, goal)
    It takes 10 steps to find a path using BFS
    >>> bfs_path
    [[0, 0], [1, 0], [2, 0], [3, 0], [3, 1]]
    '''
    ### YOUR CODE HERE ###
    path = []
    steps = 0
    found = False
    Source = Node(start[0], start[1], False)    #Start Node
    Goal = Node(goal[0], goal[1], False)
    Exp_it_row = [] #For row incremental exploration
    Exp_it_col = [] #For col incremental exploration
    Adj = []
    Adj_col = []  #List of adjacent nodes
    Adj_row = []
    row = []  #List of row co-ordinates
    col = []  #List of col co-ordinates
    Obstacle = []  #List of Obstacle nodes
    obst_free = []
    Q = []
    Q.append([Source.row, Source.col])
    distance = 0
    Parent = []
    i = 0
    k = 0
    l = 0
    x = 0

    grid_rows = len(grid)
    grid_col = len(grid[0])

    for i in range(grid_rows):
        j = 0
        rw = grid[i]
        for j in range(grid_col):
            if rw[j] == 1:
                Obstacle.append([i, j])
                j += 1

            else:
                obst_free.append([i, j])
                j += 1
        i += 1

    for node in Q:
        if node in Obstacle:
            continue
        elif node in obst_free:
            if node == goal:
                found = True
                steps += 1
                break
            elif node == [Source.row, Source.col]:

                N = Node(node[0], node[1], False)
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

            else:
                Adj.clear()
                N = Node(node[0], node[1], False)
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

            for v in Adj:
                if v not in Q:
                    V = Node(v[0], v[1], False)
                    V.parent = node
                    Parent.append([v, V.parent])
                    Q.append(v)
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
        print(f"It takes {steps} steps to find a path using BFS")
    else:
        print("No path found")
    return path, steps


def dfs(grid, start, goal):
    '''Return a path found by DFS alogirhm 
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
    >>> dfs_path, dfs_steps = dfs(grid, start, goal)
    It takes 9 steps to find a path using DFS
    >>> dfs_path
    [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2], [2, 3], [3, 3], [3, 2], [3, 1]]
    '''
    ### YOUR CODE HERE ###
    path = []
    steps = 0
    found = False
    Source = Node(start[0], start[1], False)  # Start Node
    Goal = Node(goal[0], goal[1], False)
    Node_goal = []
    Exp_it_row = []  # For row incremental exploration
    Exp_it_col = []  # For col incremental exploration
    Adj = []
    Adj_col = []  # List of adjacent nodes
    Adj_row = []
    row = []  # List of row co-ordinates
    col = []  # List of col co-ordinates
    Obstacle = []  # List of Obstacle nodes
    obst_free = []
    Q = []
    Q.append([Source.row, Source.col])
    distance = 0
    Parent = []
    visited = []
    visited.append(start)
    i = 0
    k = 0
    l = 0
    x = 0

    grid_rows = len(grid)
    grid_col = len(grid[0])

    for i in range(grid_rows):
        j = 0
        rw = grid[i]
        for j in range(grid_col):
            if rw[j] == 1:
                Obstacle.append([i, j])
                j += 1

            else:
                obst_free.append([i, j])
                j += 1
        i += 1

    for node in Q:
                if node == goal:
                    Node_goal = node
                    found = True
                    steps += 1
                    break
                elif node == [Source.row, Source.col]:
                    N = Node(node[0], node[1], False)
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
                    r = 0
                    for c in range(len(col)):
                        Adj.append([N.row, col[c]])
                        while r < len(row):
                        #for r in range(len(row)):
                            Adj.append([row[r], N.col])
                            break
                        r += 1


                    N.parent = None
                    Parent.append(N.parent)
                    steps += 1

                else:
                    Adj.clear()
                    N = Node(node[0], node[1], False)
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

                    #for c in range(len(col)):
                        #Adj.append([N.row, col[c]])
                        #for r in range(len(row)):
                            #Adj.append([row[r], N.col])
                    r = 0
                    for c in range(len(col)):
                        Adj.append([N.row, col[c]])
                        while r < len(row):
                            # for r in range(len(row)):
                            Adj.append([row[r], N.col])
                            break
                        r += 1

                    steps += 1
                Length_Q_i = len(Q)
                for a in Adj:
                    if a in Obstacle:
                        continue
                    elif a in obst_free:
                        if a not in visited:
                            Q.append(a)
                            visited.insert(0, a)
                            Parent.append([a, node])
                            break
                Length_Q_ii = len(Q)
                Change_in_Len_Q = Length_Q_ii - Length_Q_i
                if Change_in_Len_Q == 0:
                    j = 0
                    while Change_in_Len_Q == 0:
                        if (1 + j) > len(Q):
                            break
                        nd = Q[-1 - j]
                        Adj.clear()
                        N = Node(nd[0], nd[1], False)
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

                        #for c in range(len(col)):
                            #Adj.append([N.row, col[c]])
                            #for r in range(len(row)):
                                #Adj.append([row[r], N.col])

                        r = 0
                        for c in range(len(col)):
                            Adj.append([N.row, col[c]])
                            while r < len(row):
                                # for r in range(len(row)):
                                Adj.append([row[r], N.col])
                                break
                            r += 1

                        steps += 1
                        Length_Q_i = len(Q)
                        for a in Adj:
                            if a in Obstacle:
                                continue
                            elif a in obst_free:
                                if a not in visited:
                                    Q.append(a)
                                    visited.insert(0, a)
                                    Parent.append([a, nd])
                                    break
                        Length_Q_ii = len(Q)
                        Change_in_Len_Q = Length_Q_ii - Length_Q_i
                        j += 1
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
        print(f"It takes {steps} steps to find a path using DFS")
    else:
        print("No path found")
    return path, steps


# Doctest
if __name__ == "__main__":
    # load doc test
    from doctest import testmod, run_docstring_examples
    # Test all the functions
    testmod()
