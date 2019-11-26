import numpy as np
import pygame
from time import time, sleep
from random import randint as r
import random
import matplotlib.pyplot as plt

n = 4  # represents no. of side squares(n*n total squares)
reward = np.zeros((n, n))
police_position_init = [3,3]
current_pos = [0, 0]
police_position = police_position_init
reward[1, 1] = 10
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

epsilon = 0.1


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


def select_action(current_state, current_police_state):
    global current_pos, epsilon
    possible_actions = []
    action=0
    if np.random.uniform() < epsilon:
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
        vals = list()
        if current_pos[0] != 0:  # up
            vals.append(Q[current_state, current_police_state, 0])
            possible_actions.append('up')
        if current_pos[0] != n - 1:  # down
            vals.append(Q[current_state, current_police_state, 1])
            possible_actions.append('down')
        if current_pos[1] != 0:  # left
            vals.append(Q[current_state, current_police_state, 2])
            possible_actions.append('left')
        if current_pos[1] != n - 1:  # right
            vals.append(Q[current_state, current_police_state, 3])
            possible_actions.append('right')
        vals.append(Q[current_state, current_police_state, 4])
        possible_actions.append('stay')
        # action = random.choice([i for i, a in enumerate(possible_actions) if a == np.max(
        #     np.array(possible_actions))])  # randomly selecting one of all possible actions with maximin value
        max_ = random.choice([i for i, a in enumerate(vals) if a == np.max(vals)])
        action = actions[possible_actions[max_]]
    return action

def is_notpossible(current_state,current_police_state):
    vals = list()
    possible_actions = list()
    if current_pos[0] != 0:  # up
        vals.append(Q[current_state, current_police_state, 0])
        possible_actions.append('up')
    if current_pos[0] != n - 1:  # down
        vals.append(Q[current_state, current_police_state, 1])
        possible_actions.append('down')
    if current_pos[1] != 0:  # left
        vals.append(Q[current_state, current_police_state, 2])
        possible_actions.append('left')
    if current_pos[1] != n - 1:  # right
        vals.append(Q[current_state, current_police_state, 3])
        possible_actions.append('right')
    vals.append(Q[current_state, current_police_state, 4])
    possible_actions.append('stay')
    max_ = random.choice([i for i, a in enumerate(vals) if a == np.max(vals)])
    return actions[possible_actions[max_]]

def random_pos():
    A = np.zeros(5, dtype=float)
    possible_actions = 1
    if current_pos[0] != 0:
        A[actions.get("up")]=1
        possible_actions+=1
    if current_pos[0] != n - 1:
        A[actions.get("down")] = 1
        possible_actions += 1
    if current_pos[1] != 0:
        A[actions.get("left")] = 1
        possible_actions += 1
    if current_pos[1] != n - 1:
        A[actions.get("right")] = 1
        possible_actions += 1
    A[actions.get("stay")] = 1
    return A

def epsilon_greedy(state1, state2):
    AA= random_pos()
    A = AA * epsilon / np.sum(AA)
    best_action = is_notpossible(state1, state2)
    A[best_action] += (1.0 - epsilon)
    return A

def get_reward():
    global police_position, current_pos
    if current_pos == police_position:
        return -10
    else:
        return reward[current_pos[0], current_pos[1]]


def episode():
    global current_pos, epsilon, police_position, p_action
    current_robber_state = states[(current_pos[0], current_pos[1])]
    current_police_state = states[(police_position[0], police_position[1])]
    #action_probs = epsilon_greedy(current_robber_state, current_police_state)
    #n_action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
    n_action = select_action(current_robber_state,current_police_state)
    if n_action == 0:  # move up
        current_pos[0] -= 1
    elif n_action == 1:  # move down
        current_pos[0] += 1
    elif n_action == 2:  # move left
        current_pos[1] -= 1
    elif n_action == 3:  # move right
        current_pos[1] += 1

    new_state_robber = states[(current_pos[0], current_pos[1])]
    new_police_position = move_police(police_position)
    new_police_state = states[(new_police_position[0], new_police_position[1])]
    #if new_state not in terminals:
    n_arr[current_robber_state, current_police_state, p_action] += 1
    alpha = 1/pow(n_arr[current_robber_state, current_police_state, p_action], (2/3))
    Q[current_robber_state, current_police_state, p_action] += alpha * (
                    get_reward() + gamma * Q[new_state_robber,new_police_state, n_action] - Q[current_robber_state, current_police_state, p_action])
    p_action = n_action
    # if epsilon > 0.05:
    #     epsilon -= 3e-4 # reducing as time increases to satisfy Exploration & Exploitation Tradeoff


iterations = 1000000
run=0
value_func = np.zeros(iterations)
current_robber_state = states[(current_pos[0], current_pos[1])]
current_police_state = states[(police_position[0], police_position[1])]
#action_probs = epsilon_greedy(current_robber_state, current_police_state)
#p_action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

p_action = select_action(current_robber_state,current_police_state)
while run < iterations:
    # sleep(0.3)
    episode()
    value_func[run] = np.max(Q[0,15])
    run+=1
    #epsilon = epsilon + 0.5/(run+1)

plt.figure()
plt.plot(list(range(iterations)),value_func)
plt.show()




