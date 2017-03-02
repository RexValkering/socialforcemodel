import numpy as np
from itertools import count
from matplotlib.patches import Circle
from .math import *
from .area import Area
import time


class Pedestrian(object):
    """ Generate a pedestrian object.

    Generates a pedestrian and links it to the group. If start and/or
    target_path are not defined, it generates them from the spawn,
    target_path and target_area defined in the group.

    Args:
        pedestrian_id: unique id for this pedestrian
        group: the group to which this pedestrian is assigned
        desired_velocity: the desired velocity for this pedestrian
        relaxation_time: how quickly a pedestrian moves to their desired
            velocity
        start: a np.array([x,  y]) with the spawn point
        target_path: a list of np.array([x,  y]) objects that represents
            the path that this pedestrian should follow. If the pedestrian
            can see the next target in the list (meaning it is not
            obstructed from view by a wall) it will remove the previous
            target. Once the final target has been reached, the pedestrian
            will either be removed from the world or be given a new target
            within the target area.
    """
    _ids = count(0)

    def __init__(self, group=None, radius=0.15, mass=60, desired_velocity=1.3,
                 maximum_velocity=2.6, relaxation_time=2.0, start=None,
                 target_path=[]):
        self.id = self._ids.next()
        self.group = group
        self.diameter = 2 * radius
        self.radius = radius
        self.mass = float(mass)
        self.desired_velocity = desired_velocity
        self.maximum_velocity = maximum_velocity
        self.relaxation_time = relaxation_time
        self.target_is_point = True

        # Generate a spawn if not defined.
        if start is None:
            self.position = group.generate_spawn()
        else:
            self.position = start

        # Generate a target path if not defined.
        if not target_path or target_path is []:
            self.target_is_point = False
            self.target_path = group.generate_target_path()
        else:
            self.target_path = target_path

        # Start with a velocity of zero.
        velocity = self.desired_direction() * self.desired_velocity
        self.velocity = np.array([velocity[0],  velocity[1]])

        self.next_velocity = self.velocity
        self.next_position = self.position
        self.speed = np.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
        self.next_speed = self.speed

        # Make sure this pedestrian is in the group.
        if group:
            group.add_pedestrian(self)
        self.target = self.target_path[0]

        # Create a measurement history object to be filled on 'update'.
        self.measurements = []

    def add_measurement(self, category, name, value):
        """ Add a measurement for this time step to this pedestrian.

        Args:
            category: category of measurement
            name: name of measurement
            value: value of measurement
        """
        if category not in self.measurements[-1]:
            self.measurements[-1][category] = dict()
        self.measurements[-1][category][name] = value

    def get_measurement(self, category, name, time=-1):
        """ Get a measurement for a time step of this pedestrian.

        Args:
            category: category of measurement
            name: name of measurement
            time: array offset to get measurement from (default last)
        """
        if time not in self.measurements:
            return False
        return self.measurements[time][category][name]

    def step(self, step_size, obstacles):
        """ Calculate the position and velocity at next timestep.

        The new values are stored in the next_position and next_velocity
        attributes of the pedestrian. These are not immediately updated to
        avoid an order in which pedestrians are considered. Call the update
        function after step to update the position and velocity.

        Args:
            step_size: size of step, or time (dt) over which is propagated.
            pedestrians: list of pedestrians in system
            obstacles: list of obstacles in system
        """

        if len(self.target_path) == 0:
            self.next_velocity = np.array([0.0, 0.0])
            return

        self.measurements.append({})

        # Calculate the new velocity and position using Social Force
        # Propagation.
        position, velocity = self.calculate_position_and_velocity(
            step_size, obstacles)

        position_point = np.array([position[0],  position[1]])

        # Update own position and velocity.
        self.next_position = position
        self.next_velocity = velocity

        # Update barrier counts.
        for barrier in self.group.world.barriers:
            barrier.check_if_passed(self)

        self.next_speed = np.sqrt(length_squared(self.next_velocity))

        # Add position, velocity and speed to history.
        self.add_measurement('self', 'position', self.next_position)
        self.add_measurement('self', 'velocity', self.next_velocity)
        self.add_measurement('self', 'speed', self.next_speed)

    def calculate_position_and_velocity(self, step_size, obstacles):
        """ Calculate the next position and velocity if this pedestrian.

        Args:
            step_size: size of step, or time (dt) over which is propagated.
            obstacles: list of obstacles in system

        Returns:
            np.array of size 2 containing new position
            np.array of size 2 containing new velocity
        """
        force = self.calculate_force(obstacles)

        # Calculate the new velocity
        velocity_new = self.velocity + step_size * force

        # Adjust new velocity if it is above maximum_velocity
        maximum_velocity = self.maximum_velocity
        if length_squared(velocity_new) > maximum_velocity**2:
            factor = np.sqrt((velocity_new[0]**2 + velocity_new[1]**2) /
                             maximum_velocity**2)
            velocity_new = velocity_new / factor

        position_new = self.position + step_size * velocity_new

        return position_new, velocity_new

    def update(self):
        """ Update the target_path and characteristics of this pedestrian. """
        if not self.quad.inside(self.next_position):
            self.group.world.quadtree.remove(self)

        # Update position, velocity and speed.
        self.position = self.next_position
        self.velocity = self.next_velocity
        self.speed = self.next_speed

        if not self.quad:
            self.group.world.quadtree.add(self)

        # In case of a continuous domain, check if the pedestrian has
        # passed the boundaries. If so, move it.
        if self.group.world.continuous_domain:
            self.position[0] = self.position[0] % self.group.world.width
            self.position[1] = self.position[1] % self.group.world.height

        if len(self.target_path) is 0:
            return

        if len(self.target_path) > 1:
            # Look for another target that is not obstructed from view.
            obstructed = False
            while not obstructed and len(self.target_path) > 1:
                next_target = self.target_path[1]
                for obstacle in self.group.world.obstacles:
                    result = obstacle.intersects(self.position, next_target)
                    if result is not None:
                        obstructed = True
                        break

                if not obstructed:
                    self.target_path = self.target_path[1:]

        # If there is only one target and it is within the threshold range,
        # remove it from this passenger.
        threshold = self.group.world.target_distance_threshold
        if len(self.target_path) == 1:
            if self.distance_to_target() < threshold:
                self.target_path = []

        if len(self.target_path):
            target = self.target_path[0]
            self.target = target

    def arrived(self):
        """ Returns whether the pedestrian has arrived at its target. """
        if len(self.target_path) > 1:
            return False

        if len(self.target_path) > 0 and isinstance(self.target_path[0], Area):
            return self.group.target_area.in_area(self.position)

        if self.target_path == [] or self.distance_to_target() < \
                self.group.world.target_distance_threshold:
            return True

        return False

    def distance_to_target(self):
        """ Returns the distance to the current target. """
        if self.target_path == []:
            return 0.0

        target = self.target_path[0]

        if isinstance(target, Area):
            new_target = [0.0, 0.0]
            pos = self.position
            new_target[0] = min(max(pos[0], target.start[0]), target.end[0])
            new_target[1] = min(max(pos[1], target.start[1]), target.end[1])
            target = new_target

        return np.sqrt(length_squared(self.position - target))

    def set_desired_velocity(self, velocity):
        """ Set the desired velocity of this pedestrian. """
        self.desired_velocity = velocity

    def desired_direction(self):
        """ Get the desired direction of this pedestrian. """
        target = self.target_path[0]

        if isinstance(target, Area):
            new_target = [0.0, 0.0]
            pos = self.position
            new_target[0] = min(max(pos[0], target.start[0]), target.end[0])
            new_target[1] = min(max(pos[1], target.start[1]), target.end[1])
            target = new_target

        desired_dir = target - self.position
        return desired_dir / np.linalg.norm(desired_dir)

    def calculate_force(self, obstacles):
        """ Calculates the sum of attractive and repulsive forces.

        Args:
            pedestrians: list of pedestrians in system
            obstacles: list of obstacles in system

        Returns:
            np.array: sum of forces
        """

        # Get the pedestrians in the current neighbourhood.
        tree = self.group.world.quadtree
        threshold = self.group.world.interactive_distance_threshold
        pedestrians = tree.get_pedestrians_in_range(self.position, threshold)

        # Calculate the forces and return the sum of forces.
        attractive = self.calculate_attractive_force(pedestrians)
        ped_repulsive = self.calculate_pedestrian_repulsive_force(pedestrians)
        ob_repulsive = self.calculate_obstacle_repulsive_force(obstacles)

        self.add_measurement('forces', 'attractive', attractive)
        self.add_measurement('forces', 'pedestrian_repulsive', ped_repulsive)
        self.add_measurement('forces', 'obstacle_repulsive', ob_repulsive)

        return attractive + ped_repulsive + ob_repulsive

    def calculate_attractive_force(self, pedestrians):
        """ Calculates the attractive force towards the next target.

        Returns:
            np.array: attractive force towards target.
        """

        desired_dir = self.desired_direction()
        velocity_factor = self.group.world.desired_velocity_importance

        # Calculate average velocity in neighbourhood.
        average_speed = 0.0
        if len(pedestrians):
            for p in pedestrians:
                average_speed += p.speed
            average_speed /= len(pedestrians)
        else:
            velocity_factor = 1.0

        # Calculate the preferred velocity, which consists of a weighted
        # sum of the desired velocity towards target, and the average velocity.
        preferred_velocity = ((velocity_factor * self.desired_velocity *
                              desired_dir) + (1 - velocity_factor) *
                              average_speed)

        attractive_force = (- self.mass * (self.velocity -
                            preferred_velocity) / self.relaxation_time)

        return attractive_force

    def calculate_pedestrian_repulsive_force(self, pedestrians):
        """ Calculates the repulsive force with all others pedestrians.

        Args:
            pedestrians: list of pedestrians in system.

        Returns:
            np.array: sum of repulsive forces from all other pedestrians.
        """

        # world = self.group.world
        # p_position = []
        # p_velocity = []
        # p_radius = []

        # for p in pedestrians:
        #     if p == self:
        #         continue

        #     p_position.append(p.position)
        #     p_velocity.append(p.radius)
        #     p_radius.append(p.radius)

        # return calculate_pedestrian_repulsive_force(
        #     world.interactive_distance_threshold, self.position,
        #           self.velocity,
        #     self.radius, p_position, p_velocity, p_radius, world.width,
        #     world.length, world.continuous_domain)

        world = self.group.world

        # The following parameters were inspired by the Java SFM implementation
        repulsion_coefficient = world.repulsion_coefficient
        falloff_length = world.falloff_length
        body_force_constant = world.body_force_constant
        friction_force_constant = world.friction_force_constant

        distance_threshold = self.group.world.interactive_distance_threshold
        neighbourhood_distances = []
        pushing_forces = []
        friction_forces = []

        force = np.zeros(2)

        # Loop through all pedestrians.
        for p in pedestrians:
            # Skip if self.
            if p == self:
                continue

            # Calculate the distance.
            position = self.position
            difference = p.position - position

            # In case of a continuous domain, we should check if the 'wrapped'
            # distance is closer.
            if self.group.world.continuous_domain:
                if difference[0] > 0.5 * self.group.world.width:
                    difference[0] = difference[0] - self.group.world.width
                elif difference[0] < - 0.5 * self.group.world.width:
                    difference[0] = difference[0] + self.group.world.width

                if difference[1] > 0.5 * self.group.world.length:
                    difference[1] = difference[1] - self.group.world.length
                elif difference[1] < - 0.5 * self.group.world.length:
                    difference[1] = difference[1] + self.group.world.length

            distance_squared = difference[0]**2 + difference[1]**2

            # Skip if the pedestrian is too far away. This saves a significant
            # amount of time in large groups.
            if distance_squared > distance_threshold:
                continue

            distance = np.sqrt(distance_squared)
            neighbourhood_distances.append(distance)

            # Agent overlap is positive if two agents 'overlap' in space.
            agent_overlap = self.radius + p.radius - distance

            # Unit vector of the difference
            difference_direction = difference / distance

            # Find normal and tangential of difference
            normal = (position - p.position) / distance
            # normal[1] = -normal[1]
            tangential = np.array([-normal[1], normal[0]])

            social_repulsion_force = repulsion_coefficient * np.exp(
                agent_overlap / falloff_length)

            pushing_force = 0
            friction_force = np.array([0, 0])

            if agent_overlap > 0:
                # Find delta, which is a factor for friction force.
                delta = (p.velocity - self.velocity) * tangential

                pushing_force = body_force_constant * agent_overlap
                friction_force = (friction_force_constant * agent_overlap *
                                  delta * tangential)

            # Sum the forces and add to total force.
            pedestrian_force = ((social_repulsion_force + pushing_force) *
                                normal + friction_force)

            pushing_forces.append(pushing_force)
            friction_forces.append(np.sqrt(friction_force[0]**2 +
                                   friction_force[1]**2))

            force += pedestrian_force

        self.add_measurement('neighbourhood', 'distances',
                             neighbourhood_distances)
        self.add_measurement('neighbourhood', 'num_neighbours',
                             len(neighbourhood_distances))
        self.add_measurement('forces', 'pushing', pushing_forces)
        self.add_measurement('forces', 'friction', friction_forces)

        return force

    def calculate_obstacle_repulsive_force(self, obstacles):
        """ Calculates the repulsive force with the closest obstacle.

        Args:
            obstacles: list of obstacles in the system.

        Returns:
            np.array: repulsive force from closest obstacle.
        """

        # The following parameters were inspired by the Java SFM implementation
        world = self.group.world
        repulsion_coefficient = world.repulsion_coefficient
        falloff_length = world.falloff_length
        body_force_constant = world.body_force_constant
        friction_force_constant = world.friction_force_constant

        distance_threshold = 2.0

        force = np.zeros(2)

        # Loop through all obstacles.
        for o in obstacles:

            # Calculate the distance.
            position = self.position
            obstacle_position, normal = o.closest_point(self.position,
                                                        distance_threshold)
            if obstacle_position is None:
                continue

            difference = obstacle_position - position

            distance = np.sqrt(difference[0]**2 + difference[1]**2)

            # Agent overlap is positive if two agents 'overlap' in space.
            agent_overlap = self.radius - distance

            # Unit vector of the difference
            difference_direction = difference / distance

            # Find tangential
            tangential = np.array([-normal[1], normal[0]])

            obstacle_repulsion_force = repulsion_coefficient * np.exp(
                agent_overlap / falloff_length)

            pushing_force = 0
            friction_force = 0

            if agent_overlap > 0:
                # Find delta, which is a factor for friction force.
                delta = self.velocity * tangential

                pushing_force = body_force_constant * agent_overlap
                friction_force = (friction_force_constant * agent_overlap *
                                  delta * tangential)

            # Sum the forces and add to total force.
            obstacle_force = ((obstacle_repulsion_force + pushing_force) *
                              normal + friction_force)

            force += obstacle_force

        return force

    def plot(self, ax, color, add_quiver=False, plot_target=False):
        """ Plot this pedestrian on a figure.

        Args:
            ax: the pyplot axis on which to draw the pedestrian
            add_quiver: whether to add a quiver or not
            kwargs: pyplot keyword arguments
        """

        ax.add_artist(Circle(xy=(self.position), radius=0.5 * self.diameter,
                             color=color, fill=0))
        # ax.plot(self.position[0], self.position[1], marker='o',
        #         markersize=self.diameter, color=color)
        if add_quiver:
            ax.quiver(self.position[0], self.position[1],
                      self.velocity[0], self.velocity[1], angles='xy',
                      scale_units='xy', color=color)
        if plot_target:
            ax.scatter(self.target_path[0][0], self.target_path[0][1])
