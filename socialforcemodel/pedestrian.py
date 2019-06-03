import numpy as np
from collections import deque
from itertools import count
from matplotlib.patches import Circle
import matplotlib
import matplotlib.pyplot as plt
from .math import length_squared, angle_between
from .area import Area
import time
from scipy.ndimage.interpolation import rotate
from .pedestriannumba import calculate_pedestrian_repulsive_force


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

    def __init__(self, group, **kwargs):

        self.radius = 0.2
        self.mass = 60
        self.desired_velocity = 1.3
        self.maximum_velocity = 2.6
        self.relaxation_time = 0.5
        self.start = None
        self.target_path = []

        default_values = group.get_pedestrian_defaults()

        # Override default values
        for key, default_value in default_values.items():
            if key in kwargs:
                setattr(self, key, kwargs[key])
            else:
                setattr(self, key, default_value)

        if 'start' in kwargs:
            self.position = kwargs['start']
        else:
            self.position = group.generate_spawn()

        self.id = next(self._ids)
        self.group = group
        self.diameter = 2 * self.radius

        self.target_is_point = True
        self.angle_ou_process = False
        self.velocity_ou_process = False
        self.desired_offset_angle = 0.0
        self.desired_velocity_offset = 0.0
        self.closest_obstacle_points = []
        self.is_braking = False
        self.original_desired_velocity = self.desired_velocity

        self.target_index = 0
        self.target = self.target_path[self.target_index]
        self.quad = None

        self.initialize()
        
        self.current_max_repulsion = 160.0

        # Make sure this pedestrian is in the group.
        group.add_pedestrian(self)
        self.max_repulsive_force = group.world.turbulence_max_repulsion

        # Create a measurement history object to be filled on 'update'.
        self.measurements = []

    def initialize(self):
        # Start with a velocity equal to the desired velocity.
        self.desired_direction = self.get_desired_direction()
        velocity = self.desired_direction * self.desired_velocity
        self.velocity = np.array([velocity[0],  velocity[1]])

        self.next_velocity = self.velocity
        self.next_position = self.position
        self.speed = np.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
        self.next_speed = self.speed

    def add_measurement(self, category, name, value):
        """ Add a measurement for this time step to this pedestrian.

        Args:
            category: category of measurement
            name: name of measurement
            value: value of measurement
        """
        # if category != 'self':
        #     return

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
        if time is not -1 and time not in self.measurements:
            return False
        return self.measurements[time][category][name]

    def set_ornstein_uhlenbeck_process(self, mean, theta, sigma, process):
        """ Enable the Ornstein-Uhlenbeck process for deviating desired
        direction.

        The desired direction of the pedestrian will emulate brownian motion,
        but will gravitate towards the target.

        Args:
            mean: mean angle offset, usually zero
            theta: scaling factor of difference with mean
            sigma: scaling factor of random variation
        """
        if process == 'angle':
            self.angle_ou_process = (mean, theta, sigma)
        elif process == 'velocity':
            self.velocity_ou_process = (mean, theta, sigma)


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
            self.next_speed = 0.0
            return

        braking_chance = self.group.world.braking_chance
        if braking_chance > 0 and np.random.random() < braking_chance:
            # print("{}: Braking {}".format(self.group.world.time, self.id))
            self.desired_velocity = 0.0
            self.is_braking = True

        self.measurements.append({})
        self.add_measurement('self', 'time', self.group.world.time)

        # Calculate the new velocity and position using Social Force
        # Propagation.
        position, velocity = self.calculate_position_and_velocity(
            step_size, obstacles)

        # In case of a continuous domain, check if the pedestrian has
        # passed the boundaries. If so, move it.
        if self.group.world.continuous_domain:
            # Make a copy for future reference
            copy = np.array(position)
            position[0] = position[0] % self.group.world.width
            position[1] = position[1] % self.group.world.height

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

        self_position = np.array(self.position)

        position_new = self_position + step_size * velocity_new

        return position_new, velocity_new

    def goal_obstructed(self, goal, factor=1.0):
        """ Check whether the path to a goal is obstructed by obstacles. It does so by taking
        three points at consistent angles and checking if their line of sight is obstructed."""
        pos = np.array(self.position)
        if isinstance(goal, Area):
            goal = goal.get_closest_point(pos)
        
        goal_vec = goal - pos
        if length_squared(goal_vec) < 0.001:
            return False

        positions_checked = [pos + factor * self.radius * np.array([np.cos(angle), np.sin(angle)])
                             for angle in [0, 2.0 * np.pi / 3, 4.0 * np.pi / 3]]

        for obstacle in self.group.world.obstacles:
            for position in positions_checked:
                if obstacle.intersects(position, goal) is not None:
                    return True

        return False

    def update(self):
        """ Update the target_path and characteristics of this pedestrian. """
        if not self.quad.inside(self.next_position):
            self.group.world.quadtree.remove(self)
            try:
                self.group.world.quadtree.remove(self)
            except KeyError:
                print(self.position)
                print(self.next_position)
                print(self.quad.length)
                print(self.quad.xmin)
                print(self.quad.ymin)
                print(self.group.world.quadtree.length)
                raise

        # Determine whether braking should stop.
        if self.is_braking and (self.speed != self.next_speed and self.speed / self.next_speed < 1.5):     
            self.is_braking = False
            self.desired_velocity = self.original_desired_velocity

        # Update position, velocity and speed.
        self.position = self.next_position
        self.velocity = self.next_velocity
        self.speed = self.next_speed

        # Updated desired velocity if Ornstein Uhlenbeck processes are enabled.
        if self.angle_ou_process is not False:
            # mean, theta, sigma = self.ornstein_uhlenbeck

            mean, theta, sigma = self.angle_ou_process

            angle = (theta * (mean - self.desired_offset_angle) *
                     self.group.world.step_size + sigma *
                     np.random.normal())
            self.desired_offset_angle += angle

        if self.velocity_ou_process is not False:
            # mean, theta, sigma = self.ornstein_uhlenbeck

            mean, theta, sigma = self.velocity_ou_process

            diff = (theta * (mean - self.desired_velocity_offset) *
                     self.group.world.step_size + sigma *
                     np.random.normal())
            self.desired_velocity_offset += diff

        # Make sure the pedestrian is inside the quadtree.
        if not self.quad:
            self.group.world.quadtree.add(self)

        # If the target index is out of range, we have reached our destination.
        if len(self.target_path) == self.target_index:
            return

        # First check if the current target is not obstructed. If it is, work
        # back through the target path to find the closest previous target that
        # is unobstructed.
        while True:
            obstructed = self.goal_obstructed(self.target)

            if obstructed:
                if self.target_index == 0:
                    # print "Error ({}): path obstructed.".format(self.id)
                    break
                else:
                    self.target_index -= 1
                    self.target = self.target_path[self.target_index]
            else:
                break

        # Find the next target that is not obstructed.
        if len(self.target_path) - self.target_index > 1:
            # Look for another target that is not obstructed from view.
            obstructed = False
            while not obstructed and len(self.target_path) - self.target_index > 1:
                next_target = self.target_path[self.target_index + 1]
                obstructed = self.goal_obstructed(next_target, factor=1.1)

                if not obstructed:
                    self.target_index += 1
                    self.target = self.target_path[self.target_index]

        self.desired_direction = self.get_desired_direction()

        # If there is only one target and it is within the threshold range,
        # remove it from this passenger.
        threshold = self.group.world.target_distance_threshold
        if len(self.target_path) - self.target_index == 1:
            if self.distance_to_target() < threshold:
                self.target_index += 1
                self.target = None

        # if len(self.target_path) > self.target_index:
        #     target = self.target_path[self.target_index]
        #     self.target = target

    def get_desired_direction(self):
        """ Finds and returns the desired direction of this pedestrian. """
        target = self.target

        # If the target is an area, find the closest point.
        if isinstance(target, Area):
            target = target.get_closest_point(self.position)

        # Calculate the desired direction
        desired_dir = target - self.position

        # If the target equals position, return a zero vector
        if length_squared(desired_dir) < 0.001:
            # print "Warning: target equals position"
            return desired_dir

        # Add a small angle to the desired direction.
        if self.desired_offset_angle != 0.0:
            # print self.desired_offset_angle
            c = np.cos(self.desired_offset_angle)
            s = np.sin(self.desired_offset_angle)
            R = np.matrix([[c, -s],  [s, c]])
            res = R.dot(desired_dir)
            desired_dir = np.array([res[0, 0], res[0, 1]])

        return desired_dir / np.linalg.norm(desired_dir)

    def arrived(self):
        """ Returns whether the pedestrian has arrived at its target. """
        if self.target_index == len(self.target_path):
            return True

        if self.target_index + 1 < len(self.target_path):
            return False

        if isinstance(self.target, Area):
            return self.group.target_area.in_area(self.position)

        if self.distance_to_target() < self.group.world.target_distance_threshold:
            return True

        return False

    def distance_to_target(self):
        """ Returns the distance to the current target. """
        if self.target_index == len(self.target_path):
            return 0.0

        target = self.target

        if isinstance(target, Area):
            target = target.get_closest_point(self.position)

        return np.sqrt(length_squared(self.position - target))

    def set_desired_velocity(self, velocity):
        """ Set the desired velocity of this pedestrian. """
        self.desired_velocity = velocity

    def calculate_force(self, obstacles):
        """ Calculates the sum of attractive and repulsive forces.

        Args:
            pedestrians: list of pedestrians in system
            obstacles: list of obstacles in system

        Returns:
            np.array: sum of forces
        """

        # Get the pedestrians in the current neighbourhood.
        world = self.group.world
        tree = world.quadtree
        threshold = world.interactive_distance_threshold
        pedestrians = tree.get_pedestrians_in_range(self.position, threshold)

        # If we are currently in a continuous domain, we may need to find
        # neighbours on the other side of the domain.
        if world.continuous_domain:

            # For an x near the threshold
            if self.position[0] - threshold < 0:
                new_position = np.array(self.position)
                new_position[0] = world.width
                new_threshold = np.sqrt(threshold**2 - self.position[0]**2)
                pedestrians = pedestrians.union(
                    tree.get_pedestrians_in_range(new_position, new_threshold)
                )

            if self.position[0] + threshold > world.width:
                new_position = np.array(self.position)
                new_position[0] = 0
                new_threshold = np.sqrt(threshold**2 - (
                    world.width - self.position[0])**2)
                pedestrians = pedestrians.union(
                    tree.get_pedestrians_in_range(new_position, new_threshold)
                )

            # For a y near the threshold
            if self.position[1] - threshold < 0:
                new_position = np.array(self.position)
                new_position[1] = world.height
                new_threshold = np.sqrt(threshold**2 - self.position[1]**2)
                pedestrians = pedestrians.union(
                    tree.get_pedestrians_in_range(new_position, new_threshold)
                )

            if self.position[1] + threshold > world.height:
                new_position = np.array(self.position)
                new_position[1] = 0
                new_threshold = np.sqrt(threshold**2 - (
                    world.height - self.position[1])**2)
                pedestrians = pedestrians.union(
                    tree.get_pedestrians_in_range(new_position, new_threshold)
                )

        # Calculate the forces and return the sum of forces.
        attractive = self.calculate_attractive_force(pedestrians)
        ped_repulsive, ped_physical = self.calculate_pedestrian_repulsive_force(pedestrians)
        ob_repulsive = self.calculate_obstacle_repulsive_force(obstacles)

        # Ignore other pedestrians if braking.
        if self.is_braking:
            print(attractive, ped_repulsive, ob_repulsive)
            ped_repulsive = np.array([0.0, 0.0])

        total_force = attractive + self.group.repulsion_weight * (
            ped_repulsive + ob_repulsive + ped_physical)

        random_force = np.zeros(2)
        if world.velocity_variance_factor > 0.0:
            stdev = world.velocity_variance_factor * world.step_size
            random_force = np.random.normal(0.0, stdev, size=2)

        # Check whether the unit is going backwards
        target = self.target
        if isinstance(target, Area):
            target = target.get_closest_point(self.position)
        angle_between_force_and_target = angle_between(total_force, target - self.position)
        angle_between_velocity_and_target = angle_between(self.velocity, target - self.position)

        self.add_measurement('forces', 'force_angle', angle_between_force_and_target)
        self.add_measurement('forces', 'velocity_angle', angle_between_velocity_and_target)
        self.add_measurement('forces', 'attractive', attractive)
        self.add_measurement('forces', 'pedestrian_repulsive', ped_repulsive)
        self.add_measurement('forces', 'obstacle_repulsive', ob_repulsive)

        return total_force + random_force

    def calculate_attractive_force(self, pedestrians):
        """ Calculates the attractive force towards the next target.

        Returns:
            np.array: attractive force towards target.
        """

        desired_dir = self.desired_direction
        velocity_factor = self.group.world.desired_velocity_importance
        # braking_chance = self.group.world.braking_chance
        preferred_velocity = np.array([0.0, 0.0])

        # Calculate average velocity in neighbourhood.
        average_velocity = np.array([0.0, 0.0])
        if velocity_factor < 1.0 and len(pedestrians):
            for p in pedestrians:
                average_velocity += p.velocity
            average_velocity /= len(pedestrians)
        else:
            velocity_factor = 1.0

        # print(self.relaxation_time)

        # There might be a chance the pedestrian randomly brakes.
        # In that case, set preferred_velocity to (0, 0).
        # if braking_chance > 0 and random.random() < braking_chance:
        #     pass
        # else:

        # Calculate the preferred velocity, which consists of a weighted
        # sum of the desired velocity towards target and the average
        # velocity.
        actual_desired_velocity = max(0.0, self.desired_velocity + self.desired_velocity_offset)
        preferred_velocity = ((velocity_factor * actual_desired_velocity * desired_dir) + 
                              (1 - velocity_factor) * average_velocity)

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

        world = self.group.world
        p_position = []
        p_velocity = []
        p_radius = []

        for p in pedestrians:
            if p == self:
                continue

            p_position.append(p.position)
            p_velocity.append(p.radius)
            p_radius.append(p.radius)

        if len(p_position) == 0:
            self.add_measurement('forces', 'local_density', 0.0)
            self.add_measurement('forces', 'local_velocity_variance', 0.0)
            self.add_measurement('forces', 'repulsive_force', 0.0)
            self.add_measurement('forces', 'pushing_force', 0.0)
            return np.array([0.0, 0.0]), np.array([0.0, 0.0])

        force_args = []
        # force_args.append(world.turbulence_max_repulsion)
        force_args.append(self.max_repulsive_force)
        force_args.append(world.turbulence_lambda)
        force_args.append(world.turbulence_d0)
        force_args.append(world.turbulence_d1)
        force_args.append(float(world.body_force_constant))
        force_args.append(float(world.friction_force_constant))

        smoothing_squared = world.smoothing_parameter**2
        smoothing_factor = 1.0 / (np.pi * smoothing_squared)
        force_args.append(smoothing_squared)
        force_args.append(smoothing_factor)

        force = calculate_pedestrian_repulsive_force(
            world.interactive_distance_threshold, self.position, self.velocity,
            self.radius, self.speed, p_position, p_velocity, p_radius,
            world.height, world.width, world.continuous_domain,
            world.ignore_pedestrians_behind, self.desired_direction,
            force_args, world.turbulence_exponent)

        # print(force)
        self.add_measurement('forces', 'local_density', force[4])
        self.add_measurement('forces', 'local_velocity_variance',
                             force[5])
        self.add_measurement('forces', 'repulsive_force', force[6])
        self.add_measurement('forces', 'pushing_force', force[7])

        return np.array([force[0], force[1]]), np.array([force[2], force[3]])

    def calculate_pedestrian_repulsive_force_old(self, pedestrians):
        """ Calculates the repulsive force with all others pedestrians.
        This function is deprecated.

        Args:
            pedestrians: list of pedestrians in system.

        Returns:
            np.array: sum of repulsive forces from all other pedestrians.
        """
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
        local_density = 0.0
        local_velocity_variance = 0.0

        # Loop through all pedestrians.
        for p in pedestrians:
            # Skip if self.
            if p == self:
                continue

            p_position = np.array(p.position)

            # Calculate the distance.
            position = self.position
            difference = p_position - position

            # In case of a continuous domain, we should check if the 'wrapped'
            # distance is closer.
            if self.group.world.continuous_domain:
                if difference[0] > 0.5 * self.group.world.width:
                    difference[0] = difference[0] - self.group.world.width
                    p_position[0] -= self.group.world.width
                elif difference[0] < - 0.5 * self.group.world.width:
                    difference[0] = difference[0] + self.group.world.width
                    p_position[0] += self.group.world.width

                if difference[1] > 0.5 * self.group.world.height:
                    difference[1] = difference[1] - self.group.world.height
                    p_position[1] -= self.group.world.height
                elif difference[1] < - 0.5 * self.group.world.height:
                    difference[1] = difference[1] + self.group.world.height
                    p_position[1] += self.group.world.height

            distance_squared = difference[0]**2 + difference[1]**2

            # Skip if the pedestrian is too far away. This saves a significant
            # amount of time in large groups.
            if distance_squared > distance_threshold:
                continue

            # In some cases, we want to ignore pedestrians that are 'behind' us
            if self.group.world.ignore_pedestrians_behind:
                # Calculate how far 'behind' the person is.
                desired_dir = self.desired_direction
                front_factor = (desired_dir[0] * difference[0] +
                                desired_dir[1] * difference[1])
                # print "{} -- {}: {}".format(self.id, p.id, front_factor)
                # Check if the distance exceeds the sum of radi.
                if front_factor < - self.radius - p.radius:
                    continue

            distance = np.sqrt(distance_squared)
            neighbourhood_distances.append(distance)

            # Agent overlap is positive if two agents 'overlap' in space.
            agent_overlap = self.radius + p.radius - distance

            # Unit vector of the difference
            difference_direction = difference / distance

            # Find normal and tangential of difference
            normal = (position - p_position) / distance
            # normal[1] = -normal[1]
            tangential = np.array([-normal[1], normal[0]])

            # Method 1: Original
            #####################
            if not world.turbulence:
                social_repulsion_force = repulsion_coefficient * np.exp(
                    agent_overlap / falloff_length)

            # Method 2: Turbulent force
            ############################
            else:
                max_repulsive_force = world.turbulence_max_repulsion
                labda = world.turbulence_lambda
                k = world.turbulence_exponent
                D_zero = world.turbulence_d0
                D_one = world.turbulence_d1

                try:
                    factor = max(distance, 0.15)
                    cos_angle = self.desired_direction * difference_direction
                    omega = labda + (1 - labda) * (1 + cos_angle) / 2
                    social_repulsion_force = (max_repulsive_force * omega *
                                              np.exp(- factor / D_zero +
                                                     (D_one / factor)**k))
                except:
                    print(distance, cos_angle, omega)
                    raise

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
            pressure = smoothing_factor * np.exp(-distance_squared /
                                                 smoothing_squared)

            local_density += pressure
            local_velocity_variance += self.speed * pressure

        if local_density != 0:
            local_velocity_variance /= local_density

        # self.add_measurement('neighbourhood', 'distances',
        #                      neighbourhood_distances)
        # self.add_measurement('neighbourhood', 'num_neighbours',
        #                      len(neighbourhood_distances))
        # self.add_measurement('forces', 'pushing', pushing_forces)
        # self.add_measurement('forces', 'friction', friction_forces)
        self.add_measurement('forces', 'local_density', local_density)
        self.add_measurement('forces', 'local_velocity_variance',
                             local_velocity_variance)

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
        desired_dir = self.get_desired_direction()

        self.closest_obstacle_points = []

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

            self.closest_obstacle_points.append(obstacle_position)

            difference = obstacle_position - position

            distance = np.sqrt(difference[0]**2 + difference[1]**2)

            # Agent overlap is positive if two agents 'overlap' in space.
            agent_overlap = self.radius - distance

            # Unit vector of the difference
            difference_direction = difference / distance

            # Find tangential
            tangential = np.array([-normal[1], normal[0]])

            # Get angle
            # cos_angle = desired_dir * difference_direction
            # omega = world.turbulence_lambda + (1 - world.turbulence_lambda) * (1 + cos_angle) / 2
            # obstacle_repulsion_force = repulsion_coefficient * omega * np.exp(
            #     agent_overlap / falloff_length)
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

        # red_blue = cm = plt.get_cmap('RdBu')
        # c_norm = matplotlib.colors.SymLogNorm(linthresh=0.03, linscale=0.03,
        #                                       vmin=-1.0, vmax=1.0)
        # scalarmap = matplotlib.cm.ScalarMappable(norm=c_norm, cmap=red_blue)
        # core_color = scalarmap.to_rgba(self.speed - self.desired_velocity)

        c = 'white'
        # if self.speed < 0.2:
        #     c = 'red'

        ax.add_artist(Circle(xy=(self.position), radius=0.5 * self.diameter,
                             facecolor=c, edgecolor=color, fill=True))

        mirrorred_positions = []

        if self.position[0] < self.radius:
            mirrorred_positions.append(self.position + np.array([self.group.world.width, 0]))
        if self.position[0] > self.group.world.width - self.radius:
            mirrorred_positions.append(self.position - np.array([self.group.world.width, 0]))

        for position in mirrorred_positions:
            ax.add_artist(Circle(xy=position, radius=0.5 * self.diameter,
                                 facecolor=c, edgecolor=color, fill=True))

        if add_quiver:
            ax.quiver(self.position[0], self.position[1],
                      self.velocity[0], self.velocity[1], angles='xy',
                      scale_units='xy', color=color)
        if plot_target:
            target = self.target
            if isinstance(target, Area):
                target = target.get_closest_point(self.position)
            ax.scatter(target[0], target[1])
            ax.plot([self.position[0], target[0]], [self.position[1], target[1]], alpha=0.3)
            for o in self.closest_obstacle_points:
                ax.plot([self.position[0], o[0]], [self.position[1], o[1]])
