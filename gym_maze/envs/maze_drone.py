import numpy as np
from .src.maze import Maze
from .src.maze_viz import Visualizer
from collections import deque

class MazeDrone:
    
    def __init__(self, rewards_env, height=10, width=10, mode=0):
        
        """
        The following dictionary maps abstract actions from `self.action_space` 
        to the direction the drone will move in if that action is taken.
        I.e. 0 corresponds to "south", 1 to "east" etc.
        """

        #print("Create Drone")

        self._action_to_direction = {
            3: np.array([-1, 0]),   # left  / weast / esquerda
            2: np.array([0, 1]),    # up    / north / cima  
            1: np.array([1, 0]),    # right / east  / direita
            0: np.array([0, -1]),   # down  / weast / baixo
        }

        # Create a new maze
        if mode == 0:
            self.maze = Maze(height, width, algorithm = "bin_tree")
        elif mode == 1:
            self.maze = Maze(height, width, algorithm = "empty")

        # Sets the drone's position to the start of the maze
        self._drone = self.maze.entry_coor

        # Drone status flags
        self._is_done = False
        self._destroyed = False
        self._reached_target = False

        # If true, shows observations via prompt 
        self._human_render = False

        # Drone step counter in that episode
        self._step_counter = 0
        # Drone crash counter in that episode
        self._crash_counter = 0
        # Step limit for an episode
        self._step_limit = 50
        # Distance between the drone and target in the last state
        self._last_distance = 20.

        # Drone coordinate history
        self._history = deque([(1,2),(1,3),(1,4),(1,5)])


        self._rewards = {
            'destroyed': rewards_env['destroyed'],
            'stuck': rewards_env['stuck'],
            'reached': rewards_env['reached'],
            'standard': rewards_env['standard']
        }
        #print('Drone rewards:', self._rewards)

    def observe(self):
        """

        Return:
            An np.array containing information about the presence of walls around 
            the drone's current position and the distance from that position to 
            the target.
        """
        # Check for walls around the current position
        walls = self._get_walls()
        
        # Concatena com distância para o target
        r, theta = self._get_polar_distance()
        observation = walls + [r, theta]
        
        # Info for human mode
        if self._human_render: print(" D0   R1   U2   L3")
        if self._human_render: print(observation)
        
        return np.array(observation, dtype=np.float32)

    def action(self, action):
        """
        """
        if self._human_render: print('performs action:', action)
        if not self._crashed(action):
            self._crash_counter += 1
            self._drone += self._action_to_direction[action]

        if self._achieved_target():
            if self._human_render: print("Chegou no objetivo.")

        self._step_counter += 1
        
        # Limita steps 
        if self._step_counter >= self._step_limit:
            if self._human_render: print("Número máximo de steps atingido.")
            self._is_done = True

        

    # Definir uma política de recompensas eficiente
    def evaluate(self, obs):
        """ This function defines the reward policy for interactions between the
        drone and the environment.

        Arg:
            obs: the current state of the environment.

        Return:
            The reward obtained for the drone's last action.
        """
        # Distance from drone position to target
        distance = obs[-1]

        # If the drone reached the step limit 
        #if self._destroyed or self._step_counter >= self._step_limit:
        # if self._step_counter >= self._step_limit:
        #     return -100. - distance

        # If the drone crashed into the wall
        if self._destroyed:
            self._destroyed = False
            return self._rewards['destroyed']

        # If the drone is stuck in a position (flickering)
        if self._check_if_stuck():
            return self._rewards['stuck']
        
        # If the drone reached the target
        if self._reached_target:
            return self._rewards['reached']
        
        # Standard movement 
        # reward = self._last_distance - distance 
        # self._last_distance = distance
        # return reward
        return self._rewards['standard']

    @property
    def is_done(self):
        #if self._destroyed or self._reached_target:
        if self._reached_target:
            self._is_done = True
        return self._is_done

    def view(self, mode):
        """ This function is a middleman between MazeEnv.render() and 
        Visualizer.show_maze(). It feeds the Visualizer instance with 
        MazeDrone attributes.

        Return:
            A matplotlib canva, to be plotted.
        """
        vis = Visualizer(self.maze, 1, "")

        if mode == 'human':
            self._human_render = True
            _ = self.observe()

        return vis.show_maze(self._drone[1], self._drone[0], self._human_render)


    def _crashed(self, action):
        """ This function checks whether the last action moved the drone 
        towards a free space or a wall.

        Return:
            True: if the last action led to the drone hitting a wall.
            False: if the last action moved the drone to a free space.
        """
        walls = self._get_walls()
        
        # Se existe uma parede na direção do movimento = crash
        if walls[action]:
            self._destroyed = True
            if self._human_render: print("Atingiu uma parede.")
            return True
        
        return False


    def _achieved_target(self):
        """ Function checks whether the drone has reached the target or not.

        Return:
            True: If the drone's position matches the target position.
            False: If the drone's position doesn't match the target's.
        """
        if self._human_render: print(self._drone, self.maze.exit_coor)
        if np.array_equal(self._drone, np.array(self.maze.exit_coor)):
            self._reached_target = True
            return True
        
        return False
    

    def _get_walls(self) -> list:
        """ This function builds a list, whose values represent whether or
        not there is a wall on a certain side of the drone, given its position. 
        Each value in the list acts as a boolean to inform the presence (1.0) or not 
        (0.0) of a wall. The sequence of walls is: [Bottom, Right, Top, Left]

        Return:
            A list of float in which each value represents the presence (with 1.)
            or not (with 0.) of a wall around [Bottom, Right, Top, Left] the drone. 
        """
        
        # Drone's current position
        lin = self._drone[1]
        col = self._drone[0]
        
        return [1. if wall else 0. for wall in self.maze.initial_grid[lin][col].walls.values()]
    

    def _get_polar_distance(self):
        """ Function that calculates the polar distance between the
        drone position and target position.

        Return:
            Two float values representing the distance and the angle (in radians)
            between the drone and the target.
        """ 
        # Arrays with x and y coordinates of each point
        x = self.maze.exit_coor[0] - self._drone[0]
        y = self.maze.exit_coor[1] - self._drone[1]

        r = np.hypot(x, y)      # Calculate radial distance (magnitude)
        theta = np.arctan2(y, x)  # Calculate angle in radians
        return r, theta
    

    def _check_if_stuck(self):
        """
        """
        
        _ = self._history.popleft()
        self._history.append( (self._drone[0], self._drone[1]) )
        
        if self._history[0] == self._history[2] and self._history[1] == self._history[3]:
            return True
        return False    

        
    def set_rewards(self, destroyed, stuck, reached, standard):
        self._rewards['destroyed'] = destroyed
        self._rewards['stuck'] = stuck
        self._rewards['reached'] = reached
        self._rewards['standard'] = standard
         
