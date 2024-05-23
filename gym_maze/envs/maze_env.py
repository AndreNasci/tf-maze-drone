import gym
from gym import spaces 
import numpy as np
from gym_maze.envs.maze_drone import MazeDrone




class MazeEnv(gym.Env):
    # Render modes supported and framerate at which it will be rendered
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.maze_drone = MazeDrone()
        
        # Justifying action and observation space
        # Actions: north, east, south, west
        self.action_space = spaces.Discrete(4)
        # Observations: wall_north, wall_east, wall_south, wall_weast, distance_target
        low = np.array([0., 0., 0., 0., 0.], dtype=np.float32)
        high = np.array([1., 1., 1., 1., 20.], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # References for human-rendering
        self.window = None
        self.clock = None

        # Start environment condidions (maze, drone position, timer)
        

    # Resets the game, returns the first observation data from the game
    def reset(self, seed=None, options=None):
        """
        It must return a tuple of the form '(obs, info)',
        where `obs` is a observation and `info` is a dictionary 
        containing additional information
        """

        del self.maze_drone
        self.maze_drone = MazeDrone()
        obs = self.maze_drone.observe()

        return obs, {}

    # Where the action will be performed
    # Reward and and state are calculated and returned
    def step(self, action):
        
        # Moves the drone
        self.maze_drone.action(action)
        
        # Gets the environment state
        obs = self.maze_drone.observe()
        
        # Calculate reward
        reward = self.maze_drone.evaluate(obs)
        
        # Check if the maze is done
        done = self.maze_drone.is_done
        
        truncated = np.bool_(False)

        # Return step information
        return obs, reward, done, truncated, {}
        

    # Render and show the game
    def render(self):
        self.maze_drone.view()



