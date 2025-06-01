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

from resources import build_agent, TrainingSession
import pandas as pd
import gc

sys.path.append('/home/naski/Documents/dev/maze_drone_v02')
import gym_maze # Esta linha precisa estar após o PATH


def main():

    if len(sys.argv) != 4:
        print("Usage: min_combination min_run max_run")
        return -1
    
    min_combination = int(sys.argv[1])
    min_run = int(sys.argv[2])
    max_run = int(sys.argv[3])

    num_iterations = 1 # @param {type:"integer"}

    initial_collect_steps = 64  # @param {type:"integer"}
    collect_steps_per_iteration = 1 # @param {type:"integer"}
    #replay_buffer_max_length = 100000  # @param {type:"integer"}
    replay_buffer_max_length = 100  # @param {type:"integer"}

    batch_size = 64  # @param {type:"integer"}
    learning_rate = 1e-3  # @param {type:"number"}
    log_interval = 100  # @param {type:"integer"}

    num_eval_episodes = 10  # @param {type:"integer"}
    eval_interval = 100  # @param {type:"integer"}

    rewards = []
    # rewards.append({
    #     'destroyed': 0.,
    #     'stuck': - 50.,
    #     'reached': 100.,
    #     'standard': -10.
    # })
    # rewards.append({
    #     'destroyed': -60.,
    #     'stuck': - 50.,
    #     'reached': 100.,
    #     'standard': -10.
    # })
    rewards.append({
        'destroyed': -10.,
        'stuck': - 5.,
        'reached': 10.,
        'standard': -1.
    })
    # rewards.append({
    #     'destroyed': -150.,
    #     'stuck': - 50.,
    #     'reached': 100.,
    #     'standard': -10.
    # })


    # Importing custom environment
    env_name = 'maze-v0'
    env = suite_gym.load(env_name)

    # Testing
    env.reset()

    train_py_env = suite_gym.load(env_name)
    # Converts environments, originally in pure Python, to tensors (using a wrapper)
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)


    for combination in range(min_combination-1, min_combination):
        
        # Agent fully connected layer params 
        fc_layer_params = (200,) 

        for run in range(min_run-1, max_run):
            print('Combination', combination + 1, ' | Run', run + 1)

            # CREATING/RESETING THE AGENT
            agent = build_agent(fc_layer_params, env, learning_rate, train_env)
            agent.initialize()

            # GENERATE TRAINING SESSION
            session = TrainingSession(env_name, rewards[combination], agent, collect_steps_per_iteration, 
                                    num_iterations, eval_interval, replay_buffer_max_length, num_eval_episodes)
            
            # TRAINING
            step_log, returns, finished, crashed, stucked, steped, _, replay_buffer = session.train()

            # LOGGING
            df_log = pd.DataFrame({'Step': step_log, 'Average Return': returns, '% Finished': finished, 'Crash Counter': crashed, 'Stuck Counter': stucked, 'Avg Steps/Episode': steped})
            #df_log.to_csv(f"logs/01-rewards-combinations/04_comb-{combination+1}-run-{run+1}.csv", index=None, header=True)

            # CLEAR MEMORY
 
            del(session)
            del(df_log)
            gc.collect()

    print("============================================================== After first training:")

    print(agent.train_step_counter)

    checkpoint_dir = './checkpoint'
    train_checkpointer = common.Checkpointer(
        ckpt_dir=checkpoint_dir,
        max_to_keep=1,
        agent=agent,
        policy=agent.policy,
        replay_buffer=replay_buffer,
        global_step=agent.train_step_counter
    )
    
    print(agent.train_step_counter)


    train_checkpointer.initialize_or_restore()
     
    print(agent.train_step_counter)

    # global_step = tf.compat.v1.train.get_global_step()

    # print(global_step)


    for combination in range(min_combination-1, min_combination):
        
        # Agent fully connected layer params 
        fc_layer_params = (200,) 

        for run in range(min_run-1, max_run):
            print('Combination', combination + 1, ' | Run', run + 1)

            # CREATING/RESETING THE AGENT
            #agent = build_agent(fc_layer_params, env, learning_rate, train_env)
            #agent.initialize()

            # GENERATE TRAINING SESSION
            session = TrainingSession(env_name, rewards[combination], agent, collect_steps_per_iteration, 
                                    200, eval_interval, replay_buffer_max_length, num_eval_episodes)
            
            # TRAINING
            step_log, returns, finished, crashed, stucked, steped, _, replay_buffer = session.train()

            # LOGGING
            df_log = pd.DataFrame({'Step': step_log, 'Average Return': returns, '% Finished': finished, 'Crash Counter': crashed, 'Stuck Counter': stucked, 'Avg Steps/Episode': steped})
            #df_log.to_csv(f"logs/01-rewards-combinations/04_comb-{combination+1}-run-{run+1}.csv", index=None, header=True)

            # CLEAR MEMORY
 
            del(session)
            del(df_log)
            gc.collect()

    print("============================================================== After second training:")
    print(agent.train_step_counter)

    checkpoint_dir = './checkpoint'
    train_checkpointer = common.Checkpointer(
        ckpt_dir=checkpoint_dir,
        max_to_keep=1,
        agent=agent,
        policy=agent.policy,
        replay_buffer=replay_buffer,
        global_step=agent.train_step_counter
    )
    
    print(agent.train_step_counter)


    #train_checkpointer.initialize_or_restore()
    
    #print(agent.train_step_counter)



if __name__ == "__main__":
    main()