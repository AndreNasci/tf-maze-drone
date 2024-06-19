import gym
from gym import spaces 
import numpy as np
from gym_maze.envs.maze_drone import MazeDrone
from tf_agents.specs import array_spec
from typing import Optional, Union


class MazeEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    # Render modes supported and framerate at which it will be rendered
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):

        # This value defines the size of the maze
        self._maze_size = 3

        # Creates the maze 
        self.maze_drone = MazeDrone(self._maze_size, self._maze_size)
        
        # Justifying action and observation space
        # Actions: north, east, south, west
        self.action_space = spaces.Discrete(4)
        # Observations: wall_north, wall_east, wall_south, wall_weast, distance_target
        low = np.array([0., 0., 0., 0., 0.], dtype=np.float32)
        high = np.array([1., 1., 1., 1., 20.], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode


    
    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None,):
        """ Resets the environment. Creates a new instance of MazeDrone and
        returns the first environment state. 

        Return:
            The first observation data from the environment.
        """ 
        super().reset(seed=seed)

        del self.maze_drone
        self.maze_drone = MazeDrone(self._maze_size, self._maze_size)
        obs = self.maze_drone.observe()

        if not return_info:
            return obs
        else:
            return obs, {}


    
    def step(self, action):
        """ This function performs the action received, obtains a new 
        observation of the environment state, calculates the reward and checks
        if the episode is done.

        Return:
            obs: observation of the environment, i.e, four booleans indicating 
                the presence of walls around the drone's actual position and 
                a float value representing the distance to the target.
            reward: the amount of reward earned from that action. 
            done: true if the episode is completed, false otherwise. 
            {}: a dictionary available for additional info. 
        """
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
        return obs, reward, done, {'step' : self.maze_drone._step_counter}


    # Render and show the game
    def render(self, mode="rgb_array", **kwargs):
        """ This function renders and shows the actual state of the environment,
        when in human mode. Otherwise, it generates and returns the canva with 
        the content.
        
        Return:
            A matplotlib canva, to be plotted. 
        """
        return self.maze_drone.view(mode)
