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


def main():

    if len(sys.argv) != 4:
        print("Usage: X Y Z -> X = Combination | Y = Min Run | Z = Max Run")
        return -1
    
    min_combination = int(sys.argv[1])
    min_run = int(sys.argv[2])
    max_run = int(sys.argv[3])


    num_iterations = 10_000 # @param {type:"integer"}

    initial_collect_steps = 64  # @param {type:"integer"}
    collect_steps_per_iteration = 1 # @param {type:"integer"}
    #replay_buffer_max_length = 100000  # @param {type:"integer"}
    replay_buffer_max_length = 100  # @param {type:"integer"}

    batch_size = 64  # @param {type:"integer"}
    learning_rate = 1e-3  # @param {type:"number"}
    log_interval = 100  # @param {type:"integer"}

    num_eval_episodes = 10  # @param {type:"integer"}
    eval_interval = 100  # @param {type:"integer"}

    # Agent fully connected layer params 
    fc_layer_params = (200,) 

    # File's name
    description = "26"

    maze_size = 3

    rewards = []
    rewards.append({
        'destroyed': -10.,
        'stuck': - 6.,
        'reached': 10.,
        'standard': 1.
    })
    rewards.append({
        'destroyed': -10.,
        'stuck': -11.,
        'reached': 10.,
        'standard': 1.
    })
    rewards.append({
        'destroyed': -10.,
        'stuck': -15.,
        'reached': 10.,
        'standard': 1.
    })


    sys.path.append('/home/naski/Documents/dev/maze_drone_v02')
    import gym_maze # Esta linha precisa estar após o PATH

    # Importing custom environment
    env_name = 'maze-v0'
    env = suite_gym.load(env_name)

    # Testing
    env.reset()

    train_py_env = suite_gym.load(env_name)
    # Converts environments, originally in pure Python, to tensors (using a wrapper)
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)





    # CREATING/RESETING THE AGENT
    agent = build_agent(fc_layer_params, env, learning_rate, train_env)
    agent.initialize()

    # GENERATE TRAINING SESSION
    session = TrainingSession(description, maze_size, env_name, rewards[0], agent, collect_steps_per_iteration, 
                            1, eval_interval, replay_buffer_max_length, num_eval_episodes)

    # TRAINING
    step_log, returns, finished, crashed, stucked, steped, _, replay_buffer, _ = session.train()

    # CLEAR MEMORY
    del(session)



    print("============================================================== After first training:")

    print(agent.train_step_counter)

    checkpoint_dir = './checkpoint/phase-2-stuck'
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

    agent._optimizer.learning_rate = learning_rate
    print(agent._optimizer.learning_rate)

    print("========================================================================== Training:")


    for combination in range(min_combination-1, min_combination):

        for run in range(min_run-1, max_run):
            print('Combination', combination + 1, ' | Run', run + 1)
            train_checkpointer = common.Checkpointer(
                ckpt_dir=checkpoint_dir,
                max_to_keep=1,
                agent=agent,
                policy=agent.policy,
                replay_buffer=replay_buffer,
                global_step=agent.train_step_counter
            )
            agent._optimizer.learning_rate = learning_rate
            print(agent.train_step_counter)


            # GENERATE TRAINING SESSION
            session = TrainingSession(description, maze_size, env_name, rewards[combination], agent, collect_steps_per_iteration, 
                                    num_iterations, eval_interval, replay_buffer_max_length, num_eval_episodes)
            
            # TRAINING
            step_log, returns, finished, crashed, stucked, steped, log_loss, _, _ = session.train(without_wall_training=False)

            # LOGGING
            df_log = pd.DataFrame({'Step': step_log, 'Average Return': returns, '% Finished': finished, 'Crash Counter': crashed, 'Stuck Counter': stucked, 'Avg Steps/Episode': steped, 'Loss log': log_loss})
            df_log.to_csv(f"logs/03-walls/{description}_comb-{combination+1}-run-{run+1}.csv", index=None, header=True)

            # CLEAR MEMORY
            #del(agent)
            del(session)
            del(df_log)
            gc.collect()

if __name__ == "__main__":
    main()