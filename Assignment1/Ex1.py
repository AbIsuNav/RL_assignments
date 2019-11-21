import numpy as np
import maze as mz
import pandas as pd

# Description of the maze as a numpy array
maze = np.array([
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 1],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 2, 0, 0]
])

mz.draw_maze(maze)
env = mz.Maze(maze)
#env.show()
horizon = 20
V, policy = mz.dynamic_programming(env,horizon);
# Simulate the shortest path starting from position A
method = 'DynProg';
start = (0,0);
path, pathm, win = env.simulate(start, policy, method);
if win:
    print("I win!")
# Show the shortest path
mz.animate_solution(maze, path, pathm)