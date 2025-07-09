import random
import numpy as np

import gym
import gym_maze


# def simulate():
#     global epsilon, epsilon_decay
    
#     print('hello world')

#     for episode in range(MAX_EPISODES):
#         # Init environment
#         #state = env.reset()
#         total_reward = 0



if __name__ == "__main__":

    # Game mode:
    #   0 - random sample
    #   1 - user input
    GAME_MODE = 1


    env = gym.make("maze-v0")

    observation, _ = env.reset()

    action_taken = 0
    while action_taken != -1:
    


        if GAME_MODE:
            action_taken = int(input())
        else:
            action_taken = env.action_space.sample()
        print(action_taken)
        _, reward, done, _, _ = env.step(action_taken)

        print("Reward:", reward)

        if done: break

        if GAME_MODE == 0: action_taken = -1

    

    # Depends on you
    MAX_EPISODES = 10
    MAX_TRY = 5
