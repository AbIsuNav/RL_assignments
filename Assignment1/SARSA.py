import numpy as np
import pygame
from time import time, sleep
from random import randint as r
import random
import matplotlib.pyplot as plt

n = 4  # represents no. of side squares(n*n total squares)
scrx = n * 100
scry = n * 100
background = (51, 51, 51)  # used to clear screen while rendering
#screen = pygame.display.set_mode((scrx, scry))  # creating a screen using Pygame
colors = [(51, 51, 51) for i in range(n ** 2)]
reward = np.zeros((n, n))
terminals = []
penalities = 0
police_position_init = [3,3]
police_position = police_position_init
reward[1, 1] = 1
colors[5] = (0, 255, 0)
terminals.append(n ** 2 - 1)
Q = np.zeros((n ** 2, 5))# Initializing Q-Table
n_arr = np.zeros((n ** 2, 5))
actions = {"up": 0, "down": 1, "left": 2, "right": 3, "stay": 4}  # possible actions
states = {}
k = 0
for i in range(n):
    for j in range(n):
        states[(i, j)] = k
        k += 1
gamma = 0.8
current_pos = [0, 0]
epsilon = 0.2

def move_police(police_position):
    move = random.randint(0, 3)  # change 1 to 0 for complete moves
    if move == 0:  # move up
        police_position[0] -= 1
    elif move == 1:  # move down
        police_position[0] += 1
    elif move == 2:  # move left
        police_position[1] -= 1
    elif move == 3:  # move right
        police_position[1] += 1
    return police_position


def select_action(current_state):
    global current_pos, epsilon
    possible_actions = []
    if np.random.uniform() <= epsilon:
        if current_pos[0] != 0:
            possible_actions.append("up")
        if current_pos[0] != n - 1:
            possible_actions.append("down")
        if current_pos[1] != 0:
            possible_actions.append("left")
        if current_pos[1] != n - 1:
            possible_actions.append("right")
        possible_actions.append("stay")
        action = actions[possible_actions[r(0, len(possible_actions) - 1)]]
    else:
        m = np.min(Q[current_state])
        if current_pos[0] != 0:  # up
            possible_actions.append(Q[current_state, 0])
        else:
            possible_actions.append(m - 100)
        if current_pos[0] != n - 1:  # down
            possible_actions.append(Q[current_state, 1])
        else:
            possible_actions.append(m - 100)
        if current_pos[1] != 0:  # left
            possible_actions.append(Q[current_state, 2])
        else:
            possible_actions.append(m - 100)
        if current_pos[1] != n - 1:  # right
            possible_actions.append(Q[current_state, 3])
        else:
            possible_actions.append(m - 100)
        possible_actions.append(Q[current_state, 4])
        action = random.choice([i for i, a in enumerate(possible_actions) if a == max(
            possible_actions)])  # randomly selecting one of all possible actions with maximin value
    return action


def get_reward():
    global police_position, current_pos
    if current_pos == police_position:
        return -10
    else:
        return reward[current_pos[0], current_pos[1]]


def episode(action):
    global current_pos, epsilon, police_position
    current_state = states[(current_pos[0], current_pos[1])]
    new_action = select_action(current_state)
    if new_action == 0:  # move up
        current_pos[0] -= 1
    elif new_action == 1:  # move down
        current_pos[0] += 1
    elif new_action == 2:  # move left
        current_pos[1] -= 1
    elif new_action == 3:  # move right
        current_pos[1] += 1
    new_state = states[(current_pos[0], current_pos[1])]
    alpha = 1/(pow(n_arr[current_state, action]+1, 2/3))
    Q[current_state, action] += alpha * (get_reward() + gamma * (Q[new_state,new_action]) - Q[current_state, action])
    n_arr[current_state,action] += 1
    police_position = move_police(police_position)
    return new_action

#
# def layout():
#     c = 0
#     for i in range(0, scrx, 100):
#         for j in range(0, scry, 100):
#             pygame.draw.rect(screen, (255, 255, 255), (j, i, j + 100, i + 100), 0)
#             pygame.draw.rect(screen, colors[c], (j + 3, i + 3, j + 95, i + 95), 0)
#             c += 1
#             pygame.draw.circle(screen, (25, 129, 230), (current_pos[1] * 100 + 50, current_pos[0] * 100 + 50), 30, 0)


iterations = 10000000
run=0
value_func = np.zeros(iterations)
action_ = r(0, 4)
while run < iterations:
    # sleep(0.3)
    action_ = episode(action_)
    value_func[run] = np.max(Q[0])
    run += 1

plt.figure()
plt.plot(list(range(iterations)),value_func)
plt.show()
