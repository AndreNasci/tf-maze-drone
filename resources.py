# BUILD AGENT
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import sequential
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

# METRICS AND EVALUATION
from tf_agents.metrics import py_metric

# CHARTS AND VISUALIZATION
import matplotlib.pyplot as plt
import pandas as pd

# VIDEO GENERATOR
import base64
import imageio
import IPython




""" BUILD AGENT

    O uso do inicializador VarianceScaling com mode='fan_in' e 
    distribution='truncated_normal' no seu código é uma estratégia para manter
    a variância dos sinais aproximadamente constante nas camadas densas da rede.
    Crucial para garantir um treinamento eficiente e evitar problemas como o 
    vanishing gradient (gradiente evanescente) e o exploding gradient 
    (gradiente explosivo).

    QNetwork consists of a sequence of Dense layers followed by a dense layer
    with `num_actions` units to generate one q_value per available action as
    its output.
"""

# Define a helper function to create Dense layers configured with the right
# activation and kernel initializer.
def dense_layer(num_units):
    return tf.keras.layers.Dense(
        num_units,
        activation=tf.keras.activations.relu)


def build_agent(fc_layer_params, env, learning_rate, train_env, hist_env):

    action_tensor_spec = tensor_spec.from_spec(env.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1
    
    dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
    
    q_values_layer = tf.keras.layers.Dense(
        num_actions,
        activation=None)
    
    q_net = sequential.Sequential(dense_layers + [q_values_layer])

    if learning_rate:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam() # opção sem learning rate declarada

    train_step_counter = tf.Variable(0)



    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),    
        q_network=q_net,
        optimizer=optimizer,
        epsilon_greedy=0.5,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter)
    return agent 




""" METRICS AND EVALUATION

    compute_avg_return:

    Custom metrics must inherit py_metric.PyMetric.

    See also the metrics module for standard implementations of different metrics.
    https://github.com/tensorflow/agents/tree/master/tf_agents/metrics
"""

def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


def finish_counter(environment, policy, reached_end_reward, num_episodes=10):
    finished_episodes = 0
    for _ in range(num_episodes):

        time_step = environment.reset()

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
        
        if time_step.reward == reached_end_reward:
            finished_episodes += 1

    finished_percentage = finished_episodes / float(num_episodes)
    return finished_percentage

def compute_logs(environment, policy, rewards, num_episodes=10, verbose=False, hist_env=1):
    total_return = 0.0
    finished_episodes = 0
    crashes_counter = 0
    stuck_counter = 0
    step_counter = 0

    

    for _ in range(num_episodes):

        time_step = environment.reset()
        
        episode_return = 0.0

    

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            step_counter += 1
            
            if time_step.reward == rewards['destroyed']:
                crashes_counter += 1
            elif time_step.reward == rewards['stuck']:
                stuck_counter += 1
            episode_return += time_step.reward

        total_return += episode_return

        if time_step.reward == rewards['reached']:
            finished_episodes += 1

        if verbose: print()

    avg_steps = step_counter / float(num_episodes)
    avg_return = total_return / float(num_episodes)
    finished_percentage = finished_episodes / float(num_episodes)
    return avg_return.numpy()[0], finished_percentage, crashes_counter, stuck_counter, avg_steps



# Crash and Stuch status counter 
class MyMetric(py_metric.PyMetric):

    def __init__(self, reward, name="MyMetric"):
        super(MyMetric, self).__init__(name=name)
        self._count = 0
        self._reward = reward

    def call(self, trajectory):
        if trajectory.reward.numpy()[0] == self._reward:
            self._count += 1

    def result(self):
        return self._count
    
    def reset(self):
        self._count = 0



""" REPLAY BUFFER
"""

def build_buffer():
    pass




""" CHARTS AND VISUALIZATION

    Moving Average: serve para suavizar as flutuações de curto prazo em um conjunto de dados, 
    revelando a tendência subjacente e facilitando a identificação de pontos de reversão.

    plot_metric_per_iteration: some metrics are calculated repeatedly every certain amount of
    steps/iterations. This function is made to plot this kind of metrics.
"""


def plot_moving_avg(y_name, y_data, period=10, ylim=False, top_lim=0, bot_lim=0):
    y_axis_df = pd.DataFrame({y_name: y_data})
    
    y_axis_df['moving_avg'] = y_axis_df[y_name].rolling(window=period).mean()

    plt.figure(figsize=(10, 6))
    plt.plot(y_axis_df[y_name], label='Original Values', marker='o')
    plt.plot(y_axis_df['moving_avg'], label=f'Moving Average ({period} periods)', linestyle='dashed')
    plt.xlabel('Period')
    plt.ylabel(y_name)
    plt.title(f'Moving average - {y_name}')
    plt.legend()
    plt.grid(axis='y')
    if ylim:
        plt.ylim(top=top_lim, bottom=bot_lim)
    plt.show()

def plot_metric_per_iteration(num_iterations, interval, metric, y_label, ylim=False, top_lim=0, bot_lim=0):
    iterations = range(0, num_iterations, interval)
    plt.plot(iterations, metric)
    plt.ylabel(y_label)
    plt.xlabel('Iterations')
    plt.grid(axis='y')
    if ylim:
        plt.ylim(top=top_lim, bottom=bot_lim)
    plt.show()


def subplot_metric_aux(plt_ax, y_name, y_data, period=10, ylim=False, top_lim=0, bot_lim=0, hist_env=1):
    
    y_axis_df = pd.DataFrame({y_name: y_data})
    y_axis_df['moving_avg'] = y_axis_df[y_name].rolling(window=period).mean()

    if hist_env == 1: 
        plt.subplot(1, 5, plt_ax)
    else:
        plt.subplot(1, 6, plt_ax)

    plt.plot(y_axis_df[y_name], label='Original Values', marker='o')
    plt.plot(y_axis_df['moving_avg'], label=f'Mov Avg ({period} periods)', linestyle='dashed')
    plt.xlabel('Period (x 100 steps)')
    plt.ylabel(y_name)
    plt.title(f'Moving average - {y_name}')
    plt.legend()
    plt.grid(axis='y')

    if ylim:
        plt_ax.ylim(top=top_lim, bottom=bot_lim)


def plot_all_metrics(df, period=-1, hist_env=1):

    if period == -1:
        num_linhas, _ = df.shape
        period = int(num_linhas * 0.3)

    plt.figure(figsize=(24, 3))

    subplot_metric_aux(1, "Average Return", df['Average Return'], period=period, hist_env=hist_env)
    subplot_metric_aux(2, "% Finished", df['% Finished'], period=period, hist_env=hist_env)
    subplot_metric_aux(3, "Crash Counter", df['Crash Counter'], period=period, hist_env=hist_env)
    subplot_metric_aux(4, "Avg Steps/Episode", df['Avg Steps/Episode'], period=period, hist_env=hist_env)
    subplot_metric_aux(5, "Loss", df['Loss log'], period=period, hist_env=hist_env)
    if hist_env != 1:
        subplot_metric_aux(6, "Stuck Counter", df['Stuck Counter'], period=period, hist_env=hist_env)

    # Ajustar o layout
    plt.tight_layout()
    plt.show()




""" VIDEO GENERATOR
"""


def embed_mp4(filename):
    """Embeds an mp4 file in the notebook."""
    video = open(filename,'rb').read()
    b64 = base64.b64encode(video)
    tag = '''
    <video width="640" height="480" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4">
    Your browser does not support the video tag.
    </video>'''.format(b64.decode())

    return IPython.display.HTML(tag)

def create_policy_eval_video(policy, filename, eval_env, eval_py_env, num_episodes=10, fps=24):
    filename = filename + ".mp4"
    with imageio.get_writer(filename, fps=fps) as video:
        for _ in range(num_episodes):
            time_step = eval_env.reset()
            video.append_data(eval_py_env.render())
            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = eval_env.step(action_step.action)
                video.append_data(eval_py_env.render())
    return embed_mp4(filename)


""" TRAINING SESSION
"""

import pandas as pd
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.metrics import tf_metrics
import sys

sys.path.append('/home/naski/Documents/dev/maze_drone_v02')
import gym_maze # Esta linha precisa estar após o PATH

from tf_agents.drivers import dynamic_step_driver
from tf_agents.replay_buffers import tf_uniform_replay_buffer

class TrainingSession:
    
    def __init__(self, description, maze_size, env_name, rewards, agent, collect_steps_per_iteration, 
                 num_iterations, eval_interval, replay_buffer_max_length, num_eval_episodes, hist_env):
        
        self._description = description
        self._maze_size = maze_size
        self._env_name = env_name
        self._rewards = rewards
        self._agent = agent
        self._collect_steps_per_iteration = collect_steps_per_iteration
        self._num_iterations = num_iterations
        self._eval_interval = eval_interval
        self._replay_buffer_max_length = replay_buffer_max_length
        self._num_eval_episodes = num_eval_episodes
        self._hist_env = hist_env

    def reset(self):
        del(self._description)
        del(self._maze_size)
        del(self._env_name)
        del(self._rewards) 
        del(self._agent) 
        del(self._collect_steps_per_iteration) 
        del(self._num_iterations) 
        del(self._eval_interval) 
        del(self._replay_buffer_max_length) 
        del(self._num_eval_episodes)
        del(self._hist_env) 


    def train(self, without_wall_training=True, early_stop=False, verbose=False):

        # EARLY STOP
        early_stop_steps = 500

        # ENVIRONMENT
        train_py_env = suite_gym.load(self._env_name)
        eval_py_env = suite_gym.load(self._env_name)
        #wall_py_env = suite_gym.load(self._env_name)

        train_py_env.update_rewards(self._rewards['destroyed'], self._rewards['stuck'], self._rewards['reached'], self._rewards['standard'])
        eval_py_env.update_rewards(self._rewards['destroyed'], self._rewards['stuck'], self._rewards['reached'], self._rewards['standard'])
        #wall_py_env.update_rewards(self._rewards['destroyed'], self._rewards['stuck'], self._rewards['reached'], self._rewards['standard'])

        train_py_env.set_mode(int(without_wall_training))
        eval_py_env.set_mode(int(without_wall_training))

        train_py_env.set_size(self._maze_size)
        eval_py_env.set_size(self._maze_size)

        train_py_env.set_hist_env(self._hist_env)
        eval_py_env.set_hist_env(self._hist_env)

        # Converts environments, originally in pure Python, to tensors (using a wrapper)
        train_env = tf_py_environment.TFPyEnvironment(train_py_env)
        eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
        #wall_env = tf_py_environment.TFPyEnvironment(wall_py_env)

        # POLICIES
        # The main policy that is used for evaluation and deployment.
        eval_policy = self._agent.policy
        # A second policy that is used for data collection.
        collect_policy = self._agent.collect_policy

        # METRICS
        crash_counter = MyMetric(self._rewards['destroyed'])

        # REPLAY BUFFER
        replay_buffer_capacity = self._replay_buffer_max_length

        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            self._agent.collect_data_spec,
            batch_size=train_env.batch_size,
            max_length=replay_buffer_capacity)

        num_episodes = tf_metrics.NumberOfEpisodes()
        env_steps = tf_metrics.EnvironmentSteps()

        # Add an observer that adds to the replay buffer:
        replay_observer = [replay_buffer.add_batch, num_episodes, env_steps, crash_counter]


        # TRAINING THE AGENT 

        # (Optional) Optimize by wrapping some of the code in a graph using TF function.
        self._agent.train = common.function(self._agent.train)
        # Reset the train step.
        #self._agent.train_step_counter.assign(0)

        # Evaluate the agent's policy once before training.
        eval_avg_return, eval_finished_percentage, eval_crash_counter, eval_stuck_counter, eval_steps = compute_logs(eval_env, self._agent.policy, self._rewards, self._num_eval_episodes, False, self._hist_env)
        returns = [eval_avg_return]
        finished = [eval_finished_percentage]
        crashed = [eval_crash_counter]
        stucked = [eval_stuck_counter]
        steped = [eval_steps]
        step_log = [0]

        # Reset the environment.
        time_step = train_py_env.reset()
        # Set wall mode 
        # train_py_env.set_mode(int(without_wall_training))
        # eval_py_env.set_mode(int(without_wall_training))
        #wall_env.set_mode(0)

        # Create a driver to collect experience.
        collect_driver = dynamic_step_driver.DynamicStepDriver(
            train_env,
            self._agent.collect_policy,
            observers=replay_observer,
            num_steps=self._collect_steps_per_iteration)
        
        # Code needed for dataset iterator
        collect_op = dynamic_step_driver.DynamicStepDriver(
        train_env,
        self._agent.collect_policy,
        observers=replay_observer,
        num_steps=self._collect_steps_per_iteration).run()
        dataset = replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=train_env.batch_size,
            num_steps=2).prefetch(3)
        iterator = iter(dataset)


        loss_log = [0]

        steps_per_episode_log = []
        episodes_per_log_interval = []
        previous_n_episodes = 0

        crash_counter_log = []
        crash_counter_aux = 0

        avg_steps_per_episode_per_eval_interval = []

        train_py_env.print_environment()

        early_stop_counter = 0

        for _ in range(self._num_iterations):

            # Collect a few steps and save to the replay buffer.
            time_step, _ = collect_driver.run()

            dataset = replay_buffer.as_dataset(
                num_parallel_calls=3,
                sample_batch_size=train_env.batch_size,
                num_steps=2).prefetch(3)
                
            # Sample a batch of data from the buffer and update the agent's network.
            trajectories, _ = next(iterator)
            train_loss = self._agent.train(experience=trajectories).loss

            step = self._agent.train_step_counter.numpy()
            
            # Logging
            if step % 1000 == 0:
                print('step =', step)

            if step % self._eval_interval == 0:
                if verbose: print('step =', step)
                #print('step =', step)

                step_log.append(step)
                loss_log.append(float(train_loss))
                
                episodes_per_log_interval.append(num_episodes.result().numpy() - previous_n_episodes)
                steps_per_episode_log.append(self._eval_interval / (episodes_per_log_interval[-1]))
                previous_n_episodes = num_episodes.result().numpy()


                if verbose: print('  loss = {0:.2f}'.format(train_loss))

                eval_avg_return, eval_finished_percentage, eval_crash_counter, eval_stuck_counter, eval_steps = compute_logs(eval_env, self._agent.policy, self._rewards, self._num_eval_episodes, False, self._hist_env)
                returns.append(eval_avg_return)
                finished.append(eval_finished_percentage)
                crashed.append(eval_crash_counter)
                stucked.append(eval_stuck_counter)
                steped.append(eval_steps)

                # EARLY STOP
                if early_stop:
                    if early_stop == "stuck":
                        if eval_stuck_counter == 0: 
                            early_stop_counter += 1
                            #print("Early stop counter:", early_stop_counter)
                        else: 
                            early_stop_counter = 0
                            if verbose: print("Early stop reseted at", step)

                    if early_stop == "crash":
                        if eval_crash_counter == 0: 
                            early_stop_counter += 1
                            #print("Early stop counter:", early_stop_counter)
                        else: 
                            early_stop_counter = 0
                            if verbose: print("Early stop reseted at", step)

                    if early_stop == "stuckANDcrash":
                        if eval_crash_counter == 0 and eval_stuck_counter == 0:
                            early_stop_counter += 1
                            #print("Early stop counter:", early_stop_counter)
                        else: 
                            early_stop_counter = 0
                            if verbose: print("Early stop reseted at", step)



                    
                    if ( (early_stop_counter * self._eval_interval) % early_stop_steps ) and verbose:
                        eval_avg_return, eval_finished_percentage, eval_crash_counter, eval_stuck_counter, eval_steps = compute_logs(eval_env, self._agent.policy, self._rewards, 10, False, self._hist_env)
                        print("\n=====================================================================(", early_stop_counter, ")")
                        print("Small check log:")
                        print("Avg return:", eval_avg_return)
                        print("Finished:", eval_finished_percentage)
                        print("Crash Counter:", eval_crash_counter)
                        print("Stuck counter:", eval_stuck_counter)
                        print("==========================================================================\n")
                        
                    if (early_stop_counter * self._eval_interval) % early_stop_steps == 0 and early_stop_counter:
                        eval_avg_return, eval_finished_percentage, eval_crash_counter, eval_stuck_counter, eval_steps = compute_logs(eval_env, self._agent.policy, self._rewards, 100, False, self._hist_env)
                        print("\n=============================================================( EARLY STOP )")
                        print("Big check log:")
                        print("Avg return:", eval_avg_return)
                        print("Finished:", eval_finished_percentage)
                        print("Crash Counter:", eval_crash_counter)
                        print("Stuck counter:", eval_stuck_counter)
                        print("Early stop at", step)
                        print("==========================================================================\n")
                        if eval_crash_counter == 0 and eval_stuck_counter == 0 and early_stop == "stuckANDcrash":
                            break
                        elif early_stop == "crash" or early_stop == "stuck":
                            break
                        else: 
                            early_stop_counter -= 1

                if verbose: print('  Average Return = {0:.2f}'.format(eval_avg_return))
                
                if verbose: print('  Finished Percentage = {0}'.format(eval_finished_percentage))

                avg_steps_per_episode_per_eval_interval.append(self._eval_interval / sum(episodes_per_log_interval[-10:]))
                if verbose: print('  Avg of Steps/Episode: {:.2f}'.format(avg_steps_per_episode_per_eval_interval[-1]) )

                current_value = crash_counter.result()
                crash_counter_log.append(current_value)
                if verbose: print('  Crash = {0}'.format(current_value - crash_counter_aux))
                crash_counter_aux = current_value
        
        #wall_log = [compute_logs(wall_env, self._agent.policy, self._rewards, self._num_eval_episodes)]
        wall_log = 1
        print("learning_rate inside:", self._agent._optimizer.learning_rate)

        #create_policy_eval_video(self._agent.policy, f"trained-agent-{self._description}", train_env, train_py_env)

        # Clear memory
        del(dataset)
        del(trajectories)
        del(train_loss)
        del(episodes_per_log_interval)
        del(steps_per_episode_log)
        del(previous_n_episodes)
        del(avg_steps_per_episode_per_eval_interval)
        del(crash_counter_log)

        if early_stop:
            return step_log, returns, finished, crashed, stucked, steped, loss_log, replay_buffer, self._agent
        else:    
            #return step_log, returns, finished, crashed, stucked, steped, loss_log, replay_buffer, wall_log
            return step_log, returns, finished, crashed, stucked, steped, loss_log, replay_buffer, self._agent