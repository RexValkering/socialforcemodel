import numpy as np


class Pedestrian:
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

    def __init__(self, pedestrian_id, group, desired_velocity=1.5,
                 relaxation_time=2.0, start=None, target_path=[]):
        self.id = pedestrian_id
        self.group = group
        self.desired_velocity = desired_velocity
        self.relaxation_time = relaxation_time

        # Generate a spawn if not defined.
        if start is None:
            self.position = group.generate_spawn()
        else:
            self.position = start

        # Generate a target path if not defined.
        if not target_path or target_path is []:
            self.target_path = group.generate_target_path()
        else:
            self.target_path = target_path

        # Start with a velocity of zero.
        velocity = self.desired_direction() * self.desired_velocity
        self.velocity = np.array([velocity[0],  velocity[1]])

        self.next_velocity = self.velocity
        self.next_position = self.position

        # Make sure this pedestrian is in the group.
        group.add_pedestrian(self)
        self.target = self.target_path[0]

    def step(self, step_size, pedestrians, obstacles):
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
            self.passenger.velocity = [0.0, 0.0, 0.0]
            return

        # Calculate the new velocity and position using Social Force
        # Propagation.
        position, velocity = self.calculate_position_and_velocity(
            step_size, pedestrians, obstacles)

        # Adjust the velocity if it is too high.
        factor = np.linalg.norm(velocity) / self.group.world.maximum_velocity

        if factor > 1.0:
            velocity = velocity / factor

        position_point = np.array([position[0],  position[1]])

        # Check if the new position moves us through a wall.
        for obstacle in self.group.world.obstacles:
            # Calculate the intersection, if there is one.
            intersection = obstacle.intersects(self.position, position_point)
            if intersection is not None:

                # Calculate the distance to the intersection point.
                distance = np.sqrt((intersection[0] - self.position[0])**2 +
                                   (intersection[1] - self.position[1])**2)

                # Calculate the factor by which velocity should decrease.
                factor = np.linalg.norm(velocity) / (distance * step_size)
                if factor < 1.0:
                    continue

                # Decrease the velocity with a small factor more such that the
                # pedestrian does not get stuck in the wall.
                velocity = velocity / factor / 1.01

        # Update own position and velocity.
        self.next_position = position
        self.next_velocity = velocity

    def calculate_position_and_velocity(self, step_size, pedestrians,
                                        obstacles):
        """ Calculate the next position and velocity if this pedestrian.

        Args:
            step_size: size of step, or time (dt) over which is propagated.
            pedestrians: list of pedestrians in system
            obstacles: list of obstacles in system

        Returns:
            np.array of size 2 containing new position
            np.array of size 2 containing new velocity
        """
        force = self.calculate_force(pedestrians, obstacles)
        position_new = self.position + step_size * \
            self.velocity
        velocity_new = self.velocity + step_size * force

        return position_new, velocity_new

    def update(self):
        """ Update the target_path of this pedestrian. """
        self.position = self.next_position
        self.velocity = self.next_velocity

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
        if len(self.target_path) == 1:
            if self.distance_to_target() < self.group.world.distance_threshold:
                self.target_path = []

        if len(self.target_path):
            target = self.target_path[0]
            # self.pedestrian.target = [target[0], target[1], 0]
            self.target = target

    def arrived(self):
        """ Returns whether the pedestrian has arrived at its target. """
        if len(self.target_path) > 1:
            return False

        if self.target_path == [] or self.distance_to_target() < \
                self.group.world.distance_threshold:
            return True

        return False

    def distance_to_target(self):
        if self.target_path == []:
            return 0.0

        return np.linalg.norm(self.position - self.target_path[0])

    def desired_direction(self):
        """ Get the desired direction of this pedestrian. """
        target = self.target_path[0]
        desired_dir = target - self.position
        return desired_dir / np.linalg.norm(desired_dir)

    def calculate_force(self, pedestrians, obstacles):
        """ Calculates the sum of attractive and repulsive forces.

        Args:
            pedestrians: list of pedestrians in system
            obstacles: list of obstacles in system

        Returns:
            np.array: sum of forces
        """
        attractive = self.calculate_attractive_force()
        ped_repulsive = self.calculate_pedestrian_repulsive_force(pedestrians)
        ob_repulsive = self.calculate_obstacle_repulsive_force(obstacles)

        return attractive + ped_repulsive + ob_repulsive

    def calculate_attractive_force(self):
        """ Calculates the attractive force towards the next target.

        Returns:
            np.array: attractive force towards target.
        """
        desired_dir = self.desired_direction()

        # Calculate the attractive force.
        attractive_force = (1.0 / self.relaxation_time) * \
            (self.desired_velocity * desired_dir - self.velocity)
        return attractive_force

    def calculate_pedestrian_repulsive_force(self, pedestrians):
        """ Calculates the repulsive force with all others pedestrians.

        Args:
            pedestrians: list of pedestrians in system.

        Returns:
            np.array: sum of repulsive forces from all other pedestrians.
        """

        lambda_importance = 2.0     # Relative importance
        gamma = 0.35                # Speed interaction
        n = 2                       # Speed interaction
        n_prime = 3                 # Angular interaction
        force = np.zeros(2)

        # Loop through all pedestrians.
        for p in pedestrians:
            # Skip if self.
            if p == self:
                continue

            # Calculate the distance.
            position = self.position
            difference = p.position - position
            distance = np.linalg.norm(difference)

            # Skip if the pedestrian is too far away. This saves time in
            # large groups.
            if distance > 1.0:
                continue

            difference_direction = difference / distance

            p_velocity = p.velocity
            velocity_difference = self.velocity - p_velocity
            interaction = lambda_importance * velocity_difference + \
                difference_direction
            interaction_direction = interaction / np.linalg.norm(interaction)

            theta = np.arccos(np.clip(np.dot(interaction_direction,
                                             difference_direction), -1.0, 1.0))

            B = gamma * np.linalg.norm(interaction)

            velocity_amount = -np.exp(-distance / B - (n_prime * B * theta)**2)
            angle_amount = np.sign(interaction_direction) * \
                np.exp(-distance / B - (n * B * theta)**2)

            force_velocity = velocity_amount * interaction_direction
            force_angle = angle_amount * interaction_direction

            force += force_velocity * force_angle

        return force

    def calculate_obstacle_repulsive_force(self, obstacles):
        """ Calculates the repulsive force with the closest obstacle.

        Args:
            obstacles: list of obstacles in the system.

        Returns:
            np.array: repulsive force from closest obstacle.
        """

        # Find the closest point.
        min_dist = np.inf
        closest_point = None

        for o in obstacles:
            closest = o.closest_point(self.position, min_dist)
            if closest is not None:
                dist = np.linalg.norm(closest - self.position)
                if dist < min_dist:
                    min_dist = dist
                    closest_point = closest

        if closest_point is None:
            return np.zeros(2)

        diff = self.position - closest_point
        force = np.exp(- min_dist / 1.0)

        return force * (diff / np.linalg.norm(diff))
