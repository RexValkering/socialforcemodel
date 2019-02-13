from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
from .world import World
from .area import Area
from .group import Group
from .pedestrian import Pedestrian
from .obstacle import Obstacle
import numpy as np


class ParameterLoader(object):
    """ Load parameters from file and create a world. """

    def __init__(self, filename):
        data = {}
        with open(filename) as file:
            data = load(file, Loader=Loader)

        world = World()
        self.world = world

        # A dictionary for keywords with their associative function calls for
        # the SFM World.
        standard_parse_functions = {
            'world_height': world.set_height,
            'world_width': world.set_width,
            'desired_velocity': world.set_desired_velocity,
            'maximum_velocity': world.set_maximum_velocity,
            'step_size': world.set_step_size,
            'continuous_domain': world.set_wrap,
            'default_desired_velocity': world.set_desired_velocity,
            'default_maximum_velocity': world.set_maximum_velocity,
            'default_relaxation_time': world.set_relaxation_time,
            'desired_velocity_importance':
                world.set_desired_velocity_importance,
            'target_distance_threshold': world.set_target_distance_threshold,
            'interactive_distance_threshold':
                world.set_interactive_distance_threshold,
            'quadtree_threshold': world.set_quadtree_threshold,
            'repulsion_coefficient': world.set_repulsion_coefficient,
            'falloff_length': world.set_falloff_length,
            'body_force_constant': world.set_body_force_constant,
            'friction_force_constant': world.set_friction_force_constant,
            'ignore_pedestrians_behind': world.set_ignore_pedestrians_behind,
            'turbulence': world.set_turbulence,
            'velocity_variance_factor': world.set_velocity_variance_factor,
            'angle_variance_factor': world.set_angle_variance_factor,
            'turbulence_d0': world.set_turbulence_d0,
            'turbulence_d1': world.set_turbulence_d1,
            'turbulence_max_repulsion': world.set_turbulence_max_repulsion,
            'turbulence_lambda': world.set_turbulence_lambda,
            'turbulence_exponent': world.set_turbulence_exponent,
            'braking_chance': world.set_braking_chance,
            'test_repulsion_variance': world.set_test_repulsion_variance,
            'target_type': world.set_target_type,
            'reaction_delay': world.set_reaction_delay
        }

        # Parse world dimensions and characteristics
        for key in data:
            if key in standard_parse_functions:
                standard_parse_functions[key](data[key])
            elif key not in ['groups', 'obstacles']:
                print "Warning: parameter {} not recognized".format(key)

        # Parse groups
        if 'groups' in data:
            for g in data['groups']:
                self.parse_group(g)

        # Parse obstacles.
        if 'obstacles' in data:
            for o in data['obstacles']:
                world.add_obstacle(self.parse_obstacle(o))

    def parse_group(self, data):
        """ Parse the group data and return the corresponding group. """
        group = Group(len(self.world.groups), self.world)

        kwargs = dict()

        standard_parse_functions = {
            'desired_velocity': group.set_desired_velocity,
            'maximum_velocity': group.set_maximum_velocity,
            'relaxation_time': group.set_relaxation_time,
            'spawn_rate': group.set_spawn_rate,
            'start_time': group.set_start_time,
            'repulsion_weight': group.set_repulsion_weight,
            'spawn_max': group.set_spawn_max,
            'spawn_method': group.set_spawn_method,
            'radius': group.set_default_radius,
            'mass': group.set_default_mass
        }

        # Parse group characteristics
        for (key, function) in standard_parse_functions.items():
            if key in data:
                function(data[key])

        standard_variables = ['mass', 'radius', 'desired_velocity',
                              'maximum_velocity', 'relaxation_time',
                              'final_behaviour']

        for var in standard_variables:
            if var in data:
                kwargs[var] = data[var]

        # Parse spawn area.
        if 'spawn_area' in data:
            group.set_spawn_area(self.parse_area(data['spawn_area']))

        # Parse target area.
        if 'target_area' in data:
            group.set_target_area(self.parse_area(data['target_area']))

        # Parse target path
        if 'target_path' in data:
            for item in data['target_path']:
                if 'start' in item:
                    group.add_path_node(self.parse_area(item))
                else:
                    group.add_path_node(self.parse_point(item))

        # Parse pedestrians.
        if 'pedestrians' in data:
            for p in data['pedestrians']:
                group.add_pedestrian(self.parse_pedestrian(group, p))

        # Generate pedestrians.
        if 'num_pedestrians' in data:
            for i in range(int(data['num_pedestrians'])):
                group.spawn_pedestrian(**kwargs)

        return group

    def parse_pedestrian(self, group, data):
        """ Parse the pedestrian data and return the new Pedestrian. """
        kwargs = {}
        if 'start' in data:
            kwargs['start'] = self.parse_point(data['start'])
        if 'target' in data:
            kwargs['target_path'] = (group.path + [
                self.parse_point(data['target'])])

        standard_variables = ['mass', 'radius', 'desired_velocity',
                              'maximum_velocity', 'relaxation_time']
        for var in standard_variables:
            if var in data:
                kwargs[var] = data[var]

        print kwargs

        return Pedestrian(group, **kwargs)

    def parse_obstacle(self, data):
        """ Parse the obstacle data and return the new Obstacle. """
        points = []
        for p in data['points']:
            points.append(self.parse_point(p))
        return Obstacle(points)

    def parse_point(self, point):
        """ Parse and return a new point. """
        return np.array([point['x'], point['y']])

    def parse_area(self, area):
        """ Parse and return a new area. """
        start = area['start']
        end = area['end']
        return Area([start['x'], start['y']], [end['x'], end['y']])
