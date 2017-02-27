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
        self.id = group_id
        self.pedestrians = []
        self.spawn_area = spawn_area
        self.target_area = target_area
        self.world = world
        self.path = target_path
        self.final_behaviour = 'remove'
        self.default_mass = 60
        self.default_radius = 0.15

    def set_world(self, world):
        """ Set the world for this pedestrian group. """
        self.world = world

    def set_spawn_area(self, spawn_area):
        """ Set the spawn area for this group. """
        self.spawn_area = spawn_area

    def set_target_area(self, target_area):
        """ Set the target area for this group. """
        self.target_area = target_area

    def set_default_mass(self, mass):
        self.default_mass = mass

    def set_default_radius(self, radius):
        self.default_radius = radius

    def add_path_node(self, node):
        """ Append a target node to this target path. """
        self.path.append(node)

    def add_pedestrian(self, pedestrian):
        """ Add a pedestrian to this pedestrian group. """
        if pedestrian not in self.pedestrians:
            self.pedestrians.append(pedestrian)
        if not pedestrian.group:
            pedestrian.group = self

    def generate_pedestrian(self, **kwargs):
        """ Generate a pedestrian for this group. """
        Pedestrian(self, **kwargs)

    def remove_pedestrian(self, pedestrian):
        """ Remove a pedestrian from this group. """
        self.pedestrians.remove(pedestrian)

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
        to_remove = []
        for p in self.pedestrians:
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
            self.pedestrians.remove(p)
