from gym.envs.registration import register

'''
    This __init__ file adds new environments. (OpenAI Gym structure)

        
        id: it will be used in the main funcion to make an environment
    
        entry_point: the environment class
'''

register(
    id='maze-v0',
    entry_point='gym_maze.envs:MazeEnv',
    max_episode_steps=2000,
)