import gym
from gym import spaces 
import numpy as np
from gym_maze.envs.maze_drone import MazeDrone
from tf_agents.specs import array_spec
from typing import Optional, Union


class MazeEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    # Render modes supported and framerate at which it will be rendered
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, observation_type=1, render_mode=None):

        # This value defines the size of the maze
        self._maze_size = 3

        # Maze mode:
        #   0 - walls
        #   1 - no walls 
        self._mode = 0

        
        # Historical Environments
        # 1 - Observation: walls + distance, 3 rewards
        # 2 - Observation: walls + r + theta, 4 rewards
        # 3 - Observation: walls + r + theta + movement history, 4 rewards
        self._hist_env = observation_type
        

        # Default Rewards
        self._rewards = {
            'destroyed': -6.,
            'stuck': - 5.,
            'reached': 10.,
            'standard': -1.
        }

        # Creates the maze 
        self.maze_drone = MazeDrone(self._rewards, self._maze_size, self._maze_size, self._mode, self._hist_env)
        
        # Justifying action and observation space
        # Actions: north, east, south, west
        self.action_space = spaces.Discrete(4)

        # Observations: wall_north, wall_east, wall_south, wall_weast, distance_target, theta
        #               wall_north, wall_east, wall_south, wall_weast, action taken, (three movements ago)
        #               wall_north, wall_east, wall_south, wall_weast, action taken, (two movements ago)
        #               wall_north, wall_east, wall_south, wall_weast, action taken, (from last movement)
        # low = np.array([0., 0., 0., 0., 0., -3.15, 
        #                 -1, -1, -1, -1, -1, 
        #                 -1, -1, -1, -1, -1, 
        #                 -1, -1, -1, -1, -1], 
        #                 dtype=np.float32)
        
        # high = np.array([1., 1., 1., 1., 20., 3.15,
        #                  1, 1, 1, 1, 3,
        #                  1, 1, 1, 1, 3,
        #                  1, 1, 1, 1, 3], 
        #                  dtype=np.float32)
        
        #self.observation_space = self.set_hist_env(self._hist_env)
        self.set_hist_env(self._hist_env)


        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        #print("Environment Created")


    
    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None,):
        """ Resets the environment. Creates a new instance of MazeDrone and
        returns the first environment state. 

        Return:
            The first observation data from the environment.
        """ 
        super().reset(seed=7)

        #print("Reset Environment")
        #print('Rewards:', self._rewards)


        del self.maze_drone
        self.maze_drone = MazeDrone(self._rewards, self._maze_size, self._maze_size, self._mode, self._hist_env)
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
    

    def set_mode(self, mode):
        #print(f"Mode set: {mode}")
        self._mode = mode

    def set_size(self, size):
        #print(f"Maze site set: {size}x{size}")
        self._maze_size = size

    def set_hist_env(self, hist_env):

        if hist_env == 1:
            # Observations: wall_north, wall_east, wall_south, wall_weast, distance_target
            low = np.array([0., 0., 0., 0., 0.], 
                        dtype=np.float32)
            high = np.array([1., 1., 1., 1., 20.], 
                        dtype=np.float32)
        elif hist_env == 2:
            # Observations: wall_north, wall_east, wall_south, wall_weast, distance_target, theta
            low = np.array([0., 0., 0., 0., 0., -3.15], 
                        dtype=np.float32)
            high = np.array([1., 1., 1., 1., 20., 3.15], 
                        dtype=np.float32)
        else:
            # Observations: wall_north, wall_east, wall_south, wall_weast, distance_target, theta
            #               wall_north, wall_east, wall_south, wall_weast, action taken, (three movements ago)
            #               wall_north, wall_east, wall_south, wall_weast, action taken, (two movements ago)
            #               wall_north, wall_east, wall_south, wall_weast, action taken, (from last movement)
            low = np.array([0., 0., 0., 0., 0., -3.15, 
                            -1, -1, -1, -1, -1, 
                            -1, -1, -1, -1, -1, 
                            -1, -1, -1, -1, -1], 
                            dtype=np.float32)
            
            high = np.array([1., 1., 1., 1., 20., 3.15,
                            1, 1, 1, 1, 3,
                            1, 1, 1, 1, 3,
                            1, 1, 1, 1, 3], 
                            dtype=np.float32)

        self._hist_env = hist_env
        #print("Historical Environment updated: ", hist_env)
        self.observation_space = spaces.Box(low=low, high=high)
        

    def update_rewards(self, destroyed, stuck, reached, standard):
        self._rewards['destroyed'] = destroyed
        self._rewards['stuck'] = stuck
        self._rewards['reached'] = reached
        self._rewards['standard'] = standard
        #print('Rewards Updated:', self._rewards)

    def print_rewards(self):
        print('Rewards:', self._rewards)

    def print_environment(self):
        print("Rewards:", self._rewards)
        print("Size:", self._maze_size)
        print("Mode:", self._mode)
        print("Hist env:", self._hist_env)