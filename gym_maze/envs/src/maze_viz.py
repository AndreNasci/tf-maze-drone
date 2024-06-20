import matplotlib
import matplotlib.pyplot as plt
import logging
from matplotlib.patches import Rectangle
import numpy as np

#logging.basicConfig(level=logging.DEBUG)


class Visualizer(object):
    """Class that handles all aspects of visualization.


    Attributes:
        maze: The maze that will be visualized
        cell_size (int): How large the cells will be in the plots
        height (int): The height of the maze
        width (int): The width of the maze
        ax: The axes for the plot
        lines:
        squares:
        media_filename (string): The name of the animations and images

    """
    def __init__(self, maze, cell_size, media_filename):
        self.maze = maze
        self.cell_size = cell_size
        self.height = maze.num_rows * cell_size
        self.width = maze.num_cols * cell_size
        self.ax = None
        self.lines = dict()
        self.squares = dict()
        self.media_filename = media_filename

    def set_media_filename(self, filename):
        """Sets the filename of the media
            Args:
                filename (string): The name of the media
        """
        self.media_filename = filename

    def show_maze(self, drone_lin, drone_col, human_render):
        """Displays a plot of the maze without the solution path"""

        # Create the plot figure and style the axes
        fig = self.configure_plot()

        # Plot the walls on the figure
        self.plot_walls()

        # Plot the drone 
        self.plot_drone(drone_lin, drone_col)

        # Display the plot to the user
        if (human_render):
            plt.show()
            return None
        else:
            plt.draw()
            # Get the canvas as a NumPy array
            canvas = np.array(fig.canvas.renderer.buffer_rgba())
            

        # Handle any potential saving
        if self.media_filename:
            fig.savefig("{}{}.png".format(self.media_filename, "_generation"), frameon=None)

        plt.close(fig)

        return canvas

    def plot_walls(self):
        """ Plots the walls of a maze. This is used when generating the maze image"""
        for i in range(self.maze.num_rows):
            for j in range(self.maze.num_cols):
                if self.maze.initial_grid[i][j].is_entry_exit == "entry":
                    self.ax.text(i*self.cell_size, j*self.cell_size, "START", fontsize=7, weight="bold")
                elif self.maze.initial_grid[i][j].is_entry_exit == "exit":
                    self.ax.text(i*self.cell_size, j*self.cell_size, "END", fontsize=7, weight="bold")
                if self.maze.initial_grid[i][j].walls["top"]:
                    self.ax.plot([j*self.cell_size, (j+1)*self.cell_size],
                                 [i*self.cell_size, i*self.cell_size], color="k")
                if self.maze.initial_grid[i][j].walls["right"]:
                    self.ax.plot([(j+1)*self.cell_size, (j+1)*self.cell_size],
                                 [i*self.cell_size, (i+1)*self.cell_size], color="k")
                if self.maze.initial_grid[i][j].walls["bottom"]:
                    self.ax.plot([(j+1)*self.cell_size, j*self.cell_size],
                                 [(i+1)*self.cell_size, (i+1)*self.cell_size], color="k")
                if self.maze.initial_grid[i][j].walls["left"]:
                    self.ax.plot([j*self.cell_size, j*self.cell_size],
                                 [(i+1)*self.cell_size, i*self.cell_size], color="k")
                    
    def plot_drone(self, drone_lin, drone_col):
        
        # Drone parameters 
        side_length = self.cell_size * 0.6
        center_y = drone_lin * self.cell_size + self.cell_size / 2.0
        center_x = drone_col * self.cell_size + self.cell_size / 2.0

        # Calculate corner coordinates (for the Rectangle)
        bottom_left_x = center_x - side_length / 2
        bottom_left_y = center_y - side_length / 2

        # Create the square (as a Rectangle)
        square = Rectangle((bottom_left_x, bottom_left_y), side_length, side_length)

        # Add the square to the axes
        self.ax.add_patch(square)


    def configure_plot(self):
        """Sets the initial properties of the maze plot. Also creates the plot and axes"""

        # Create the plot figure
        fig = plt.figure(figsize = (7, 7*self.maze.num_rows/self.maze.num_cols))

        # Create the axes
        self.ax = plt.axes()

        # Set an equal aspect ratio
        self.ax.set_aspect("equal")

        # Remove the axes from the figure
        self.ax.axes.get_xaxis().set_visible(False)
        self.ax.axes.get_yaxis().set_visible(False)

        title_box = self.ax.text(0, self.maze.num_rows + self.cell_size + 0.1,
                            r"{}$\times${}".format(self.maze.num_rows, self.maze.num_cols),
                            bbox={"facecolor": "gray", "alpha": 0.5, "pad": 4}, fontname="serif", fontsize=15)

        return fig