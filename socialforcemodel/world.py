import matplotlib as mpl
import matplotlib.pyplot as plt
import random


class World:
    """ Create a new world for the simulation.

    Create a new world for pedestrians and obstacles.

    Args:
        length: length (meters in y) of this world
        width: width (meters in x) of this world
        obstacles: list of obstacles
        groups: list of pedestrian groups
        desired_velocity: the default velocity of pedestrians
        maximum_velocity: the maximum velocity of pedestrians.
            Pedestrians may go faster than their desired velocity to
            avoid collission.
        step_size: step size in seconds to advance each step.
        distance_threshold: the maximum distance between a pedestrian and
            its target for the target to be considered reached. Defaults
            to desired_velocity * step_size / 2.
    """

    def __init__(self, length, width, obstacles=[], groups=[],
                 desired_velocity=1.5, maximum_velocity=3.0,
                 step_size=0.1, distance_threshold=None):
        self.length = length
        self.width = width
        self.obstacles = obstacles
        self.groups = groups
        self.desired_velocity = desired_velocity
        self.maximum_velocity = maximum_velocity
        self.step_size = step_size
        if distance_threshold:
            self.distance_threshold = distance_threshold
        else:
            self.distance_threshold = desired_velocity * self.step_size
        self.figure = None
        self.ax = None

    def length(self):
        """ Get the length (y-direction) of this world. """
        return self.length

    def width(self):
        """ Get the width (x-direction) of this world. """
        return self.width

    def add_group(self, group):
        """ Add a group to this world. """
        if group not in self.groups:
            self.groups.append(group)
        group.world = self

    def add_obstacle(self, obstacle):
        """ Add an obstacle to this world. """
        if obstacle not in self.obstacles:
            self.obstacles.append(obstacle)

    def step(self):
        """ Advance all pedestrians in this world. """
        pedestrians = []

        if not self.figure:
            self.figure = plt.figure()
            self.ax = self.figure.add_subplot(1, 1, 1)

        # Update all groups and their pedestrian targets first.
        for group in self.groups:
            pedestrians += group.pedestrians

        if pedestrians == []:
            return False

        # Shuffle all pedestrians.
        random.shuffle(pedestrians)

        # Loop through all pedestrians in the shuffled order.
        for p in pedestrians:
            p.step(self.step_size, pedestrians, self.obstacles)

        for group in self.groups:
            group.update()

        return True

    def plot(self):

        if not self.figure:
            self.figure = plt.figure()
            self.ax = self.figure.add_subplot(1, 1, 1)

        self.ax.set_xlim([0, self.length])
        self.ax.set_ylim([0, self.width])

        colors = [c['color'] for c in list(mpl.rcParams['axes.prop_cycle'])]

        group = None
        for group in self.groups:
            for p in group.pedestrians:
                self.ax.quiver(p.position[0], p.position[1],
                               10 * p.velocity[0] * self.step_size,
                               10 * p.velocity[1] * self.step_size,
                               angles='xy', scale_units='xy',
                               scale=1, color=colors[group.id * 2])
                self.ax.scatter([p.position[0]], [p.position[1]],
                                color=colors[group.id * 2])
                self.ax.plot(p.target[0], p.target[1], 'o',
                             color=colors[group.id * 2 + 1])

        for obstacle in self.obstacles:
            arr = obstacle.pairs()
            if len(arr) == 0:
                self.ax.scatter([obstacle.points[0][0]],
                                [obstacle.points[0][1]],
                                color=colors[group.id * 2 + 2])
            else:
                X = []
                Y = []
                for point in arr[0]:
                    X.append(point[0])
                    Y.append(point[1])
                self.ax.plot(X, Y, color=colors[group.id * 2 + 2])

        figure = self.figure
        self.figure = None
        self.ax = None

        return figure
