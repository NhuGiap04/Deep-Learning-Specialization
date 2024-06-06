'''Simulated Annealing Algorithm
- Instead of picking the best move, it picks a random move
- If the move improves the situation, it is always accepted
- Otherwise, the algorithm accepts the move with probability < 1. The probability ~ exponentially with the "badness"
 of the move: -delta(E) by which the evaluation is worsened
- The probabilities also decreases as T(temperature) goes down, "bad" moves are more likely to be allowed at the
  start, when T is high, and they become unlikely as T decreases.
- If the schedule lowers T slowly enough, the algorithm will find a global optimum with probability -> 1
'''


import random
import numpy as np


def problem():
    # TODO: Implement your problem function here
    # initial state
    pass


def schedule(t):
    # TODO: Implement your scheduling function here
    # schedule: a mapping from time to "temperature"
    pass


def make_node(state):
    # return the initial node
    pass


def child_node(state, actions):
    # Input: a state and a list of available actions
    # return a list of child states
    pass


def simulated_annealing(problem, schedule, initial_temperature, action_list):
    # Initialize the root node
    current = make_node(problem.initial_state)
    while True:
        # Decrease the temperature by using schedule function
        temperature = schedule(initial_temperature)
        if temperature == 0:
            return current
        # randomly selected successor of current node
        next_node = random.choice(child_node(current, action_list))
        delta = next_node.value - current.value
        if delta > 0:
            current = next_node
        else:
            if np.random.random_sample() < np.exp(delta / temperature, np.e):
                # if next_node.value < current_node.value
                # we will assign current node = next node with probability = e^(-delta / temperature)
                # When temperature = 0 => The algorithm will be Greedy Algorithm (always goes downhill(if the objective
                # minimized)
                current = next_node


