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
terminals.append(5)
Q = np.zeros((n ** 2, n**2, 5))# Initializing Q-Table
n_arr = np.zeros((n ** 2, n**2,5))
actions = {"up": 0, "down": 1, "left": 2, "right": 3, "stay": 4}  # possible actions
states = {}
k = 0
for i in range(n):
    for j in range(n):
        states[(i, j)] = k
        k += 1
gamma = 0.8
current_pos = [0, 0]
epsilon = 0.25


def move_police(police_position):
    move = random.randint(0, 3)  # change 1 to 0 for complete moves
    if move == 0 :  # move up
        if police_position[0] == 0:
            move_police(police_position)
        else :
            police_position[0] -= 1
    elif move == 1:  # move down
        if police_position[0] == 3:
            move_police(police_position)
        else:
            police_position[0] += 1
    elif move == 2:  # move left
        if police_position[1] == 0:
            move_police(police_position)
        else:
            police_position[1] -= 1
    elif move == 3:  # move right
        if police_position[1] == 3:
            move_police(police_position)
        else:
            police_position[1] += 1
    return police_position


def select_action():
    global current_pos, epsilon
    possible_actions = []
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
    return action


def get_reward():
    global police_position, current_pos
    if current_pos == police_position:
        return -10
    else:
        return reward[current_pos[0], current_pos[1]]


def episode():
    global current_pos, epsilon, police_position
    current_robber_state = states[(current_pos[0], current_pos[1])]
    current_police_state = states[(police_position[0], police_position[1])]
    action = select_action()
    if action == 0:  # move up
        current_pos[0] -= 1
    elif action == 1:  # move down
        current_pos[0] += 1
    elif action == 2:  # move left
        current_pos[1] -= 1
    elif action == 3:  # move right
        current_pos[1] += 1

    new_state_robber = states[(current_pos[0], current_pos[1])]
    new_police_position = move_police(police_position)
    new_police_state = states[(new_police_position[0], new_police_position[1])]
    #if new_state not in terminals:
    n_arr[current_robber_state, current_police_state, action] += 1
    alpha = 1/pow(n_arr[current_robber_state, current_police_state, action]+1, 2/3)
    Q[current_robber_state, current_police_state, action] += alpha * (
                    get_reward() + gamma * (np.max(Q[new_state_robber,new_police_state])) - Q[current_robber_state, current_police_state, action])

    #    return True
    # else:
    #     Q[current_state, action] += alpha * (get_reward() - Q[current_state, action])
    #     current_pos = [0, 0]
    #     police_position = police_position_init
    #     #if epsilon > 0.05:
    #     #    epsilon -= 3e-4  # reducing as time increases to satisfy Exploration & Exploitation Tradeoff
    #     return False


# def layout():
#     c = 0
#     for i in range(0, scrx, 100):
#         for j in range(0, scry, 100):
#             pygame.draw.rect(screen, (255, 255, 255), (j, i, j + 100, i + 100), 0)
#             pygame.draw.rect(screen, colors[c], (j + 3, i + 3, j + 95, i + 95), 0)
#             c += 1
#             pygame.draw.circle(screen, (25, 129, 230), (current_pos[1] * 100 + 50, current_pos[0] * 100 + 50), 30, 0)


iterations = 1000000
run=0
value_func = np.zeros(iterations)
while run < iterations:
    # sleep(0.3)
    episode()
    #print(run)
    value_func[run] = np.max(Q[0,15])
    run+=1

plt.figure()
plt.plot(list(range(iterations)),value_func)
plt.show()




