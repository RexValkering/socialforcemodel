from .pedestrian import Pedestrian
import numpy as np


class Group(object):
    """ Class defining a group of pedestrians with a common spawn and target.

    All pedestrians in a group expose similar behaviour. They have a common
    spawn area, target area, final behaviour and target path.

    Args:
        group_id: unique id of group.
        world: world object to add this group to.
        spawn_area: Area object that defines where pedestrians spawn.
        target_area: Area object that defines where targets are set.
        target_path: Array of points defining the path taken by pedestrians.
    """

    def __init__(self, group_id, world=None, spawn_area=None,
                 target_area=None, target_path=[]):
        # Group attributes
        self.world = world
        self.id = group_id
        self.spawn_rate = 0.0
        self.spawn_area = spawn_area
        self.target_area = target_area
        self.path = target_path
        self.start_time = 0
        self.active = False

        # Pedestrian attributes
        self.pedestrians = []
        self.final_behaviour = 'remove'
        self.default_mass = 60
        self.default_radius = 0.15
        self.desired_velocity = self.world.desired_velocity
        self.maximum_velocity = self.world.maximum_velocity
        self.relaxation_time = self.world.relaxation_time

        world.add_group(self)

    def set_world(self, world):
        """ Set the world for this pedestrian group. """
        self.world = world

    def set_start_time(self, start_time):
        self.start_time = start_time

    def set_spawn_area(self, spawn_area):
        """ Set the spawn area for this group. """
        self.spawn_area = spawn_area

    def set_spawn_rate(self, spawn_rate):
        """ Set the spawn rate for this group.

        Args:
            spawn_rate: pedestrians per second.
        """
        self.spawn_rate = spawn_rate

    def set_target_area(self, target_area):
        """ Set the target area for this group. """
        self.target_area = target_area

    def set_default_mass(self, mass):
        self.default_mass = mass

    def set_default_radius(self, radius):
        self.default_radius = radius

    def set_desired_velocity(self, desired_velocity):
        self.desired_velocity = desired_velocity

    def set_maximum_velocity(self, maximum_velocity):
        self.maximum_velocity = maximum_velocity

    def set_relaxation_time(self, relaxation_time):
        self.relaxation_time = relaxation_time

    def add_path_node(self, node):
        """ Append a target node to this target path. """
        self.path.append(node)

    def add_pedestrian(self, pedestrian):
        """ Add a pedestrian to this pedestrian group. """
        if pedestrian not in self.pedestrians:
            self.pedestrians.append(pedestrian)
            if self.active:
                self.world.quadtree.add(pedestrian)

        if not pedestrian.group:
            pedestrian.group = self

    def get_pedestrians(self):
        if not self.active:
            return []
        return self.pedestrians

    def spawn_pedestrian(self, **kwargs):
        """ Generate a pedestrian for this group. """

        defaults = {'mass': self.default_mass,
                    'radius': self.default_radius,
                    'desired_velocity': self.desired_velocity,
                    'maximum_velocity': self.maximum_velocity,
                    'relaxation_time': self.relaxation_time}

        for var in defaults:
            if var not in kwargs:
                kwargs[var] = defaults[var]

        p = Pedestrian(self, **kwargs)
        if self.active:
            self.world.quadtree.add(p)

    def activate(self):
        self.active = True
        for pedestrian in self.get_pedestrians():
            self.world.quadtree.add(pedestrian)

    def remove_pedestrian(self, pedestrian):
        """ Remove a pedestrian from this group. """
        self.pedestrians.remove(pedestrian)
        self.world.quadtree.remove(pedestrian)

    def set_final_behaviour(self, behaviour):
        """ Set the final behaviour of this pedestrian group.

        Using this function you can set the final behaviour of pedestrians once
        they reach their target. The behaviour can be either 'remove', 'wander'
        or 'none':
        - On 'remove', the pedestrian is removed from the group and will no
            longer be part of the simulation.
        - On 'wander', the pedestrian will be assigned a new target within the
            target area each time it reaches its target.
        - On 'none', the target will not move after reaching its target.

        The default behavour of a group is 'remove'.

        Args:
            behaviour: either 'remove', 'wander' or 'none'.

        Raises:
            InvalidArgumentException: If group name is unknown.
        """
        if behaviour not in ['remove', 'wander', 'none']:
            raise InvalidArgumentException

        self.final_behaviour = behaviour

    def generate_spawn(self):
        """ Generate a x,y position in the spawn area. """
        if not self.spawn_area:
            raise Exception(
                'Trying to generate spawn without setting spawn area.')

        spawn = np.random.rand(2)
        area = self.spawn_area
        position = np.array([spawn[0] * area.width() + area.start[0],
                             spawn[1] * area.height() + area.start[1]])
        return position

    def generate_target(self):
        """ Generate a x,y position in the target area. """
        if not self.target_area:
            raise Exception(
                'Trying to generate target without setting target area.')
            exit(1)

        # Testing the use of an area rather than a point.
        return self.target_area

        target = np.random.rand(2)
        area = self.target_area
        position = np.array([target[0] * area.width() + area.start[0],
                            target[1] * area.height() + area.start[1]])
        return position

    def generate_target_path(self):
        """ Generate a target and return it appended to the target path. """
        return self.path + [self.generate_target()]

    def update(self):
        """ Update all pedestrian targets.

        Loop through all pedestrians and update their position, velocity and
        target. If the pedestrian has arrived at its final target, the function
        handles this according to the set final behaviour (default remove).
        """
        if not self.active:
            if self.world.time >= self.start_time:
                self.activate()
            else:
                return

        to_remove = []
        for p in self.get_pedestrians():
            # Update the pedestrians status and target.
            p.update()

            # If the pedestrian has arrived at its final target, handle
            # this according to set final behaviour.
            if p.arrived():
                if self.final_behaviour is "remove":
                    to_remove.append(p)
                    continue
                elif self.final_behaviour is "wander":
                    p.target_path.append(self.generate_target())
                else:
                    pass

        # Remove all pedestrians that can be removed from the simulation.
        for p in to_remove:
            self.remove_pedestrian(p)

        # Spawn new pedestrians based on the spawn rate.
        if self.spawn_rate:
            poisson_lambda = self.spawn_rate * self.world.step_size
            for s in range(np.random.poisson(poisson_lambda)):
                self.spawn_pedestrian()

