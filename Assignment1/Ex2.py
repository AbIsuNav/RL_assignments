import numpy as np
import maze as mz
import pandas as pd
import matplotlib.pyplot as plt

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
gamma = 29.0/30.0
epsilon = 0.0001
V, policy = mz.value_iteration(env, gamma, epsilon);
# Simulate the shortest path starting from position A
method = 'ValIter';
start = (0, 0);
path, pathm, win = env.simulate(start, policy, method);

mz.animate_solution(maze, path, pathm)