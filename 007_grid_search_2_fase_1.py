from __future__ import absolute_import, division, print_function

import os
# Keep using keras-2 (tf-keras) rather than keras-3 (keras).
os.environ['TF_USE_LEGACY_KERAS'] = '1'


import matplotlib.pyplot as plt
import numpy as np
import PIL.Image

import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.metrics import tf_metrics
from tf_agents.utils import common

import sys
import gym

from resources import build_agent, TrainingSession, compute_logs
import pandas as pd
import gc



def main():

    if len(sys.argv) != 5:
        print("Usage: X Y -> X = Combination | Y = Gamma | Z = Epsilon | A = buffer")
        return -1
    
    combination = int(sys.argv[1])
    gamma_option = int(sys.argv[2])
    epsilon_option = int(sys.argv[3])
    buffer_option = int(sys.argv[4])
    

    buffer_options = [100_000, 50_000, 10_000, 1_000, 100]
    fc_options = [(100,50), (100,50,50), (200,100), (200,)]

    gamma_options = [0.99, 0.9, 0.75, 0.5]
    eps_greedy_otions = [0.7, 0.5, 0.3]

    # Historical Environments
    # 1 - Observation: walls + distance, 3 rewards
    # 2 - Observation: walls + r + theta, 4 rewards
    # 3 - Observation: walls + r + theta + movement history, 4 rewards
    hist_env = 2


    # Hiperparameters
    num_iterations = 10_000 

    initial_collect_steps = 100
    collect_steps_per_iteration = 1 
    replay_buffer_max_length = buffer_options[buffer_option]  

    batch_size = 64  
    learning_rate = 1e-3  
    log_interval = 100  

    num_eval_episodes = 10  
    eval_interval = 100  

    # Agent fully connected layer params 
    fc_layer_params = fc_options[3]
    without_wall_training = True
    early_stop = "finished"

    # Agent hyperparameters
    gamma = gamma_options[gamma_option]
    epsilon = eps_greedy_otions[epsilon_option]

    # File's name
    description = "007"

    # Size of the maze
    maze_size = 3

    # Reward combination
    #combination = 2 
    run = 1

    rewards = []
    rewards.append({
        'destroyed': -10.,
        'stuck': - 6.,
        'reached': 10.,
        'standard': -1.
    })
    rewards.append({
        'destroyed': -10.,
        'stuck': -11.,
        'reached': 10.,
        'standard': -1.
    })
    rewards.append({
        'destroyed': -10.,
        'stuck': -15.,
        'reached': 10.,
        'standard': -1.
    })
    rewards.append({
        'destroyed': -500.,
        'stuck': 0.,
        'reached': 500.,
        'standard': -10.
    })
    rewards.append({
        'destroyed': 0.,
        'stuck': 0.,
        'reached': 100.,
        'standard': -10.
    })
    rewards.append({
        'destroyed': -50.,
        'stuck': 0.,
        'reached': 300.,
        'standard': -1.
    })


    sys.path.append('/home/naski/Documents/dev/maze_drone_v02')
    import gym_maze # Esta linha precisa estar após o PATH

    # Importing custom environment
    env_name = f'maze-v0-{hist_env}'
    env = suite_gym.load(env_name)

    # Testing
    env.reset()

    train_py_env = suite_gym.load(env_name)
    # Converts environments, originally in pure Python, to tensors (using a wrapper)
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)

    
    print("\n========================================================================== Training PHASE 1:")
    # Essa fase consiste em treinar o agente a se movimentar em um ambiente sem paredes em direção ao target, sem colidir com as paredes. 


    # CREATING/RESETING THE AGENT
    agent = build_agent(fc_layer_params, env, learning_rate, train_env, hist_env, epsilon, gamma)
    agent.initialize()


    # GENERATE TRAINING SESSION
    session = TrainingSession(description, maze_size, env_name, rewards[combination], agent, collect_steps_per_iteration, 
                            num_iterations, eval_interval, replay_buffer_max_length, num_eval_episodes, hist_env)

    # TRAINING
    step_log, returns, finished, crashed, stucked, steped, log_loss, _, agent = session.train(without_wall_training=without_wall_training, early_stop=early_stop)


    # LOGGING
    df_log_1 = pd.DataFrame({'Step': step_log, 'Average Return': returns, '% Finished': finished, 'Crash Counter': crashed, 'Stuck Counter': stucked, 'Avg Steps/Episode': steped, 'Loss log': log_loss})
    df_log_1.to_csv(f"logs/06-hist-env/{description}-combination_{combination}-epsilon_{epsilon_option}-gamma_{gamma_option}-buffer_{buffer_option}.csv", index=None, header=True)


    # print("\n========================================================================== Training PHASE 2:")
    # # Essa fase consiste em treinar o agente a se movimentar em um ambiente com paredes, sem colidir ou ficar preso.  

    # check_1 = agent.train_step_counter.numpy()

    # # Atualiza hiperparametros


    # # GENERATE TRAINING SESSION
    # session = TrainingSession(description, maze_size, env_name, rewards[combination], agent, collect_steps_per_iteration, 
    #                         num_iterations, eval_interval, replay_buffer_max_length, num_eval_episodes, hist_env)

    # # TRAINING
    # step_log, returns, finished, crashed, stucked, steped, log_loss, _, agent = session.train(without_wall_training=True, early_stop=early_stop)


    # # LOGGING
    # df_log_2 = pd.DataFrame({'Step': step_log, 'Average Return': returns, '% Finished': finished, 'Crash Counter': crashed, 'Stuck Counter': stucked, 'Avg Steps/Episode': steped, 'Loss log': log_loss})
    

    # df_log = pd.concat([df_log_1, df_log_2])
    # df_log.to_csv(f"logs/06-hist-env/{description}-combination_{combination}-epsilon_{epsilon_option}-gamma_{gamma_option}.csv", index=None, header=True)

    # print("Fase 1 terminada em", check_1)



if __name__ == "__main__":
    main()