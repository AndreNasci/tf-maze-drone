
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

import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import sequential
from tf_agents.specs import tensor_spec
from tf_agents.utils import common



# Define a helper function to create Dense layers configured with the right
# activation and kernel initializer.
def dense_layer(num_units):
    return tf.keras.layers.Dense(
        num_units,
        activation=tf.keras.activations.relu)


def build_agent(fc_layer_params, env, learning_rate, train_env):

    action_tensor_spec = tensor_spec.from_spec(env.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1
    
    dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
    
    q_values_layer = tf.keras.layers.Dense(
        num_actions,
        activation=None)
    
    q_net = sequential.Sequential(dense_layers + [q_values_layer])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

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
from tf_agents.metrics import py_metric

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


class MyMetric(py_metric.PyMetric):

    def __init__(self, name="MyMetric"):
        super(MyMetric, self).__init__(name=name)
        self._count = 0

    def call(self, trajectory):
        if trajectory.reward.numpy()[0] == -6.:
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

import matplotlib.pyplot as plt
import pandas as pd

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




""" VIDEO GENERATOR
"""

import base64
import imageio
import IPython


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

