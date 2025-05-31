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
        print("Usage: X Y -> X = Combination | Y = Checkpoint | Z = steps_per_iteration | log num")
        return -1
    
    min_combination = int(sys.argv[1])
    checkpoint_num = int(sys.argv[2])
    
    min_run = 1
    max_run = 1


    num_iterations = 50_000 # @param {type:"integer"}

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
    fc_layer_params = (200,100,50,) 

    # File's name
    description = "7000"

    maze_size = 5

    rewards = []
    rewards.append({
        'destroyed': -10.,
        'stuck': - 5.,
        'reached': 10.,
        'standard': -1.
    })
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


    logs_runs = []


    for run in range(1):

        print("\n========================================================================== Training PHASE 1:")
        # Essa fase consiste em treinar o agente a se movimentar em um ambiente sem paredes em direção ao target, sem colidir com as paredes. 


        # CREATING/RESETING THE AGENT
        agent = build_agent(fc_layer_params, env, learning_rate, train_env)
        agent.initialize()


        combination = 0 # do arquivo 05_train_nd_save_agent
        collect_steps_per_iteration = 1


        # GENERATE TRAINING SESSION
        session = TrainingSession(description, maze_size, env_name, rewards[combination], agent, collect_steps_per_iteration, 
                                num_iterations, eval_interval, replay_buffer_max_length, num_eval_episodes)
        
        # TRAINING
        step_log, returns, finished, crashed, stucked, steped, log_loss, _, agent = session.train(without_wall_training=True, early_stop="crash")


        # LOGGING
        df_log_phase_1 = pd.DataFrame({'Step': step_log, 'Average Return': returns, '% Finished': finished, 'Crash Counter': crashed, 'Stuck Counter': stucked, 'Avg Steps/Episode': steped, 'Loss log': log_loss})
        


        print("\n========================================================================== Training PHASE 2:")
        # Essa fase consiste em treinar o agente a se movimentar em um ambiente sem paredes sem ficar preso. 

        check_1 = agent.train_step_counter.numpy()

        combination = 3 # do arquivo 19_saving_checkpoint_2
        collect_steps_per_iteration = 1


        # GENERATE TRAINING SESSION
        session = TrainingSession(description, maze_size, env_name, rewards[combination], agent, collect_steps_per_iteration, 
                                num_iterations, eval_interval, replay_buffer_max_length, num_eval_episodes)
        
        # TRAINING
        step_log, returns, finished, crashed, stucked, steped, log_loss, _, agent = session.train(without_wall_training=True, early_stop="stuckANDcrash")


        # LOGGING
        df_log_phase_2 = pd.DataFrame({'Step': step_log, 'Average Return': returns, '% Finished': finished, 'Crash Counter': crashed, 'Stuck Counter': stucked, 'Avg Steps/Episode': steped, 'Loss log': log_loss})



        print("\n========================================================================== Training PHASE 3:")
        # Essa fase consiste em treinar o agente a se movimentar em um ambiente com paredes, sem colidir ou ficar preso. 

        check_2 = agent.train_step_counter.numpy()

        combination = min_combination # da série de arquivos 2000 e 3000
        collect_steps_per_iteration = int(sys.argv[3])


        # GENERATE TRAINING SESSION
        session = TrainingSession(description, maze_size, env_name, rewards[combination], agent, collect_steps_per_iteration, 
                                num_iterations, eval_interval, replay_buffer_max_length, num_eval_episodes)
        
        # TRAINING
        step_log, returns, finished, crashed, stucked, steped, log_loss, _, agent = session.train(without_wall_training=False, early_stop="stuckANDcrash")


        # LOGGING
        df_log_phase_3 = pd.DataFrame({'Step': step_log, 'Average Return': returns, '% Finished': finished, 'Crash Counter': crashed, 'Stuck Counter': stucked, 'Avg Steps/Episode': steped, 'Loss log': log_loss})



        print("===================================================================================== Testing:")

        check_3 = agent.train_step_counter.numpy()

        eval_py_env = suite_gym.load(env_name)
        eval_py_env.update_rewards(rewards[combination]['destroyed'], rewards[combination]['stuck'], rewards[combination]['reached'], rewards[combination]['standard'])
        eval_py_env.set_mode(0) # walls
        eval_py_env.set_size(maze_size)
        eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

        avg_return, finished_percentage, crashes_counter, stuck_counter, avg_steps = compute_logs(eval_env, agent.policy, rewards[combination], 1000)
        print(avg_return, finished_percentage, crashes_counter, stuck_counter, avg_steps)

        logs_runs.append({
            'run': run+1,
            'check_1': check_1,
            'check_2': check_2,
            'check_3': check_3,
            'avg_return': avg_return,
            'finished_percentage': finished_percentage,
            'crashes_counter': crashes_counter,
            'stuck_counter': stuck_counter,
            'avg_steps': avg_steps
        })

        # CREATING CHECKPOINT
        print("\n============================================================================== Creating log:")
        
        df_log = pd.concat([df_log_phase_1, df_log_phase_2, df_log_phase_3])
        df_log.to_csv(f"logs/05-validating-reprod/{description}-checkpoint_{checkpoint_num}-run_{run+1}-log_{int(sys.argv[4])}.csv", index=None, header=True)


        # CLEAR MEMORY
        del(agent)
        del(session)
        del(df_log)
        gc.collect()

    df = pd.DataFrame(logs_runs)
    print(df)
    df.to_csv(f"logs/05-validating-reprod/resumo-{description}-checkpoint_{checkpoint_num}-run_{run+1}-log_{int(sys.argv[4])}.csv", index=None, header=True)
    

if __name__ == "__main__":
    main()