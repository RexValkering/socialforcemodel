import matplotlib as mpl
import matplotlib.pyplot as plt
import random
import numpy as np
from .quadtree import QuadTree


class World(object):
    """ Create a new world for the simulation.

    Create a new world for pedestrians and obstacles.

    Args:
        height: height (meters in y) of this world
        width: width (meters in x) of this world
        desired_velocity: the default velocity of pedestrians
        maximum_velocity: the maximum velocity of pedestrians.
            Pedestrians may go faster than their desired velocity to
            avoid collission.
        step_size: step size in seconds to advance each step.
        distance_threshold: the maximum distance between a pedestrian and
            its target for the target to be considered reached. Defaults
            to desired_velocity * step_size / 2.
    """

    def __init__(self):
        # Initial world parameters
        self.height = 10
        self.width = 10
        self.step_size = 0.01
        self.time = 0.0

        # State properties
        self.started = False
        self.quadtree = None
        self.obstacles = []
        self.groups = []
        self.barriers = []
        self.measurement_functions = []
        self.measurements = []

        # Global variables
        self.desired_velocity = 1.3
        self.maximum_velocity = 2.6
        self.relaxation_time = 2.0
        self.continuous_domain = False
        self.ignore_pedestrians_behind = False
        self.desired_velocity_importance = 1.0
        self.target_distance_threshold = self.maximum_velocity * self.step_size
        self.interactive_distance_threshold = 2.0
        self.quadtree_threshold = 10
        self.repulsion_coefficient = 2000
        self.falloff_length = 0.08
        self.body_force_constant = 12000
        self.friction_force_constant = 24000
        self.turbulence = True
        self.velocity_variance_factor = 0.0
        self.angle_variance_factor = 0.0
        self.smoothing_parameter = np.sqrt(10)
        self.turbulence_d0 = 0.31
        self.turbulence_d1 = 0.45
        self.turbulence_max_repulsion = 160.0
        self.turbulence_lambda = 0.25
        self.turbulence_exponent = 2
        self.test_repulsion_variance = False
        self.target_type = 'area'
        self.reaction_delay = 0.0

        # Random braking experiment
        self.braking_chance = 0.0

    def set_height(self, height):
        self.height = height

    def set_width(self, width):
        self.width = width

    def set_desired_velocity(self, desired_velocity):
        self.desired_velocity = desired_velocity

    def set_maximum_velocity(self, maximum_velocity):
        self.maximum_velocity = maximum_velocity

    def set_relaxation_time(self, relaxation_time):
        self.relaxation_time = relaxation_time

    def set_wrap(self, value):
        self.continuous_domain = value

    def set_desired_velocity_importance(self, value):
        self.desired_velocity_importance = value

    def set_target_distance_threshold(self, value):
        self.target_distance_threshold = value

    def set_interactive_distance_threshold(self, value):
        self.interactive_distance_threshold = value

    def set_step_size(self, value):
        self.step_size = value

    def set_repulsion_coefficient(self, repulsion_coefficient):
        self.repulsion_coefficient = repulsion_coefficient

    def set_falloff_length(self, falloff_length):
        self.falloff_length = falloff_length

    def set_body_force_constant(self, body_force_constant):
        self.body_force_constant = body_force_constant

    def set_friction_force_constant(self, friction_force_constant):
        self.friction_force_constant = friction_force_constant

    def set_quadtree_threshold(self, value):
        self.quadtree_threshold = value

    def set_ignore_pedestrians_behind(self, value):
        self.ignore_pedestrians_behind = value

    def set_turbulence(self, value):
        self.turbulence = value

    def set_velocity_variance_factor(self, value):
        self.velocity_variance_factor = value

    def set_angle_variance_factor(self, value):
        self.angle_variance_factor = value

    def set_turbulence_d0(self, value):
        self.turbulence_d0 = value

    def set_turbulence_d1(self, value):
        self.turbulence_d1 = value

    def set_turbulence_max_repulsion(self, value):
        self.turbulence_max_repulsion = value

    def set_turbulence_lambda(self, value):
        self.turbulence_lambda = value

    def set_turbulence_exponent(self, value):
        self.turbulence_exponent = value

    def set_braking_chance(self, value):
        self.braking_chance = value

    def set_test_repulsion_variance(self, value):
        self.test_repulsion_variance = value

    def set_target_type(self, value):
        self.target_type = value

    def set_reaction_delay(self, value):
        self.reaction_delay = value

    def clear(self):
        for group in self.groups:
            group.clear()

    def add_group(self, group):
        """ Add a group to this world. """
        if group not in self.groups:
            self.groups.append(group)

    def add_obstacle(self, obstacle):
        """ Add an obstacle to this world. """
        if obstacle not in self.obstacles:
            self.obstacles.append(obstacle)

    def add_barrier(self, barrier):
        """ Add a measurement barrier to this world. """
        if barrier not in self.barriers:
            self.barriers.append(barrier)

    def add_measurement(self, function):
        """ Add a measurement function to this world. """
        self.measurement_functions.append(function)
        self.measurements.append([])

    def update(self):
        """ Update the current world. """
        if not self.started:
            self.initialize_tree()

        for group in self.groups:
            group.update()

    def initialize_tree(self):
        """ Initialize the Quad Tree implementation. """
        self.quadtree = QuadTree(0.0, 0.0, max(self.width, self.height),
                                 self.quadtree_threshold)
        # for group in self.groups:
        #     for pedestrian in group.get_pedestrians():
        #         self.quadtree.add(pedestrian)

        self.started = True

    def step(self):
        """ Advance all pedestrians in this world. """
        if not self.started:
            self.update()

        pedestrians = []
        expecting_spawn = False

        for group in self.groups:
            if group.spawn_rate > 0:
                expecting_spawn = True
            pedestrians += group.get_pedestrians()

        if pedestrians != [] or expecting_spawn:
            # Loop through all pedestrians in the shuffled order.
            for p in pedestrians:
                p.step(self.step_size, self.obstacles)

            self.update()

        for index in range(len(self.measurement_functions)):
            function = self.measurement_functions[index]
            self.measurements[index].append(function(self))

        self.time += self.step_size

        if pedestrians == []:
            return expecting_spawn

        return True

    def plot(self, add_quiver=False):
        """ Create a plot of the current world. """

        # Create a new plot and figure.
        plt.style.use('ggplot')
        figure = plt.figure(figsize=(17, 17))
        ax = figure.add_subplot(1, 1, 1)

        # Scale plot to current world.
        ax.set_xlim([0, self.width])
        ax.set_ylim([0, self.height])
        ax.set_aspect('equal')

        # Set colours.
        colors = [c['color'] for c in list(mpl.rcParams['axes.prop_cycle'])]

        if self.quadtree:
            self.quadtree.draw(ax)

        group = None
        # Plot all pedestrians as quivers and their targets as points.
        for group in self.groups:
            for p in group.get_pedestrians():
                p.plot(ax, color=colors[group.id % len(colors)],
                       add_quiver=add_quiver, plot_target=False)

        # Plot all obstacles as lines.
        for obstacle in self.obstacles:
            arr = obstacle.pairs()
            last_color = None

            if len(arr) == 0:
                ax.scatter([obstacle.points[0][0]],
                           [obstacle.points[0][1]],
                           color='black')
            else:
                for start, end in arr:
                    X = [start[0], end[0]]
                    Y = [start[1], end[1]]
                    # print X, Y
                    ax.plot(X, Y, color='black')

        # Return the figure.
        return figure

    def density_plot(self, tile_size=2, threshold=5):
        """ Create a density plot of the current world.
            Args:
                tile_size  height and width of density plot tiles
                threshold  threshold value for which the tile is red
        """
        # Create a new plot and figure.
        figure = plt.figure()
        ax = figure.add_subplot(1, 1, 1)

        # Create colormap and luminance map.
        norm = mpl.colors.Normalize(vmin=0, vmax=threshold)

        # Create a list of tiles.
        tile_list = np.zeros((np.ceil(float(self.height) / tile_size),
                             np.ceil(float(self.width) / tile_size)))

        tile_dim = tile_list.shape

        X, Y = np.meshgrid(np.arange(tile_dim[1] + 1) * tile_size,
                           np.arange(tile_dim[0] + 1) * tile_size)

        # Loop through all pedestrians and add them to the list of tiles.
        for group in self.groups:
            for p in group.pedestrians:
                index = np.floor(p.position / tile_size)
                tile_list[index[0]][index[1]] += 1
                ax.plot(p.position[0], p.position[1], 'o')

        ax.pcolormesh(Y, X, tile_list, rasterized=True, norm=norm, alpha=0.5)
        return figure
