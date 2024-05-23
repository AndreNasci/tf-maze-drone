import numpy as np
from .src.maze import Maze
from .src.maze_viz import Visualizer

class MazeDrone:
    
    def __init__(self, height=10, width=10):
        print('make environment')

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction the drone will move in if that action is taken.
        I.e. 0 corresponds to "north", 1 to "east" etc.
        """
        # Check this 
        self._action_to_direction = {
            0: np.array([-1, 0]),   # south
            1: np.array([0, 1]),    # east
            2: np.array([1, 0]),    # north
            3: np.array([0, -1]),   # weast
        }

        # Cria novo labirinto
        self.maze = Maze(height, width, algorithm = "bin_tree")

        # Posiciona drone na entrada do labirinto
        self._drone = self.maze.entry_coor

        # Flags de estado do drone / labirinto
        self._is_done = False
        self.destroyed = False
        self.reached_target = False

        # Plotar imagem do labirinto
        #self.view()


    def observe(self):
        
        # Check for walls around the current position
        walls = self._get_walls()
        
        # Concatena com distância para o target
        observation = walls + [self._get_target_distance()]
        
        print(" B0   R1   T2   L3")
        print(observation)
        
        return np.array(observation, dtype=np.float32)

    def action(self, action):
        
        print('performs action:')
        if not self._crashed(action):
            self._drone += self._action_to_direction[action]

        if self._achieved_target():
            print("Chegou no objetivo.")
        print(self._drone)


    def evaluate(self, obs):
        distance = obs[-1]

        if self.destroyed:
            return -500.
        
        if self.reached_target:
            return 500.
        
        return 20. - distance

    @property
    def is_done(self):
        if self.destroyed or self.reached_target:
            self._is_done = True
        return self._is_done

    def view(self):
        vis = Visualizer(self.maze, 1, "")
        vis.show_maze()

    def _crashed(self, action):
        walls = self._get_walls()
        # Se existe uma parede na direção do movimento = crash
        if walls[action]:
            self.destroyed = True
            print("Atingiu uma parede.")
            return True
        
        
        return False

    def _achieved_target(self):
        """ Function checks whether the drone has reached the target or not.

        Return:
            True: If the drone's position matches the target position.
            False: If the drone's position doesn't match the target's.
        """
        print(self._drone, self.maze.exit_coor)
        if np.array_equal(self._drone, np.array(self.maze.exit_coor)):
            self.reached_target = True
            return True
        
        return False
    

    def _get_walls(self) -> list:
        """ This function builds a list, whose values represent whether or
        not there is a wall on a certain side of the drone, given its position. 
        Each value in the list acts as a boolean to inform the presence (1.0) or not 
        (0.0) of a wall. The sequence of walls is: [Bottom, Right, Top, Right]

        Return:
            A list of float in which each value represents the presence (with 1.)
            or not (with 0.) of a wall around [Bottom, Right, Top, Right] the drone. 
        """
        
        # Drone's current position
        lin = self._drone[0]
        col = self._drone[1]
        
        return [1. if wall else 0. for wall in self.maze.initial_grid[lin][col].walls.values()]
    

    def _get_target_distance(self):
        """ Function that calculates the euclidian distance between the
        drone position and target position.

        Return:
            A float value representing the distance between the drone
            and the target.
        """ 
        return np.linalg.norm(np.array(self._drone) - np.array(self.maze.exit_coor))