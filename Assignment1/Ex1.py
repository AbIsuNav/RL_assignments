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
win_count = np.zeros(30)
#env.show()
iterations = 100
points = 30
for i in range(1,points+1):
    horizon = i
   # print("horizon = ", i)
    for _ in range(iterations):
        V, policy = mz.dynamic_programming(env, horizon);
        # Simulate the shortest path starting from position A
        method = 'DynProg';
        start = (0, 0);
        path, pathm, win = env.simulate(start, policy, method);
        if win:
            win_count[i-1] += 1
    print(i, win_count[i-1])
# Show the shortest path
# mz.animate_solution(maze, path, pathm)
prob = win_count/iterations

plt.figure()
plt.plot(list(range(1,points+1)),prob)
plt.show()