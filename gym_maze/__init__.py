from gym.envs.registration import register

'''
    This __init__ file adds new environments. (OpenAI Gym structure)

        
        id: it will be used in the main funcion to make an environment
    
        entry_point: the environment class
'''

register(
    id='maze-v0-1',
    entry_point='gym_maze.envs:MazeEnv',
    kwargs={'observation_type': 1},
    max_episode_steps=2000,
)

register(
    id='maze-v0-2',
    entry_point='gym_maze.envs:MazeEnv',
    kwargs={'observation_type': 2},
    max_episode_steps=2000,
)

register(
    id='maze-v0-3',
    entry_point='gym_maze.envs:MazeEnv',
    kwargs={'observation_type': 3},
    max_episode_steps=2000,
)