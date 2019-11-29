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
gamma = 1 - (1 / 30.0)
epsilon = 0.01
method = 'ValIter';
win_count = 0.0
start = (0, 0);
runs = 100
for _ in range(runs):
    V, policy = mz.value_iteration(env, gamma, epsilon);
    # Simulate the shortest path starting from position A
    path, pathm, win = env.simulate(start, policy, method);
    if win:
        win_count += 1.0

prob = win_count/runs
print("Final probability: ", prob)