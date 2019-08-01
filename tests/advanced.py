"""
This file contains some advanced functionality for running the experiments.
Measurement data is placed in the storage folder.
"""

import os
import csv
import math
from queue import Queue

import numpy as np
import matplotlib.pyplot as plt

import socialforcemodel as sfm

from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

import progressbar

np.seterr(all='raise')


class Experiment(object):

    def __init__(self, args):
        self.storage_folder = args.storage
        self.measurement_file = "{}/{}_measurements.csv".format(args.storage, args.outfile)
        self.pedestrian_file = "{}/{}_pedestrians.csv".format(args.storage, args.outfile)
        self.pedestrian_load_file = args.pedestrian_file
        self.all_pedestrians = set()
        self.processed_pedestrians = set()
        self.debug = args.debug
        self.image_target = os.path.join(args.storage, "img", args.outfile)
        self.image = args.image
        self.parameter_file = args.file
        self.max_pedestrians = args.max_pedestrians
        self.spawn_mode = args.spawn_mode
        self.steps = args.steps
        self.world = None
        self.labda = args.labda
        self.resume = args.resume
        self.force = args.force

    def write_intermediary_results(self):
        """Write intermediary simulation results to measurements and pedestrians."""

        # Get a set of current pedestrians
        current_pedestrians = set()
        for group in self.world.groups:
            current_pedestrians = current_pedestrians.union(group.get_pedestrians())

        # Write all pedestrian data to file.
        with open(self.pedestrian_file, "a") as ped_outfile:
            with open(self.measurement_file, "a") as csv_outfile:
                ped_writer = csv.writer(ped_outfile)
                csv_writer = csv.writer(csv_outfile)

                for pedestrian in self.all_pedestrians.union(current_pedestrians):

                    # Write to ped file
                    if pedestrian.id not in self.processed_pedestrians:
                        row = [pedestrian.id,
                               "%.4f" % pedestrian.mass,
                               "%.4f" % pedestrian.radius,
                               "%.4f" % pedestrian.desired_velocity,
                               "%.4f" % pedestrian.maximum_velocity,
                               pedestrian.group.id]
                        ped_writer.writerow(row)

                        # Set pedestrian as 'processed'
                        self.processed_pedestrians.add(pedestrian.id)

                    measurements = pedestrian.measurements
                    for arr in measurements:
                        self_measurements = arr['self']
                        row = ["%.2f" % self_measurements['time'],
                               pedestrian.id,
                               "%.4f" % self_measurements['position'][0],
                               "%.4f" % self_measurements['position'][1],
                               "%.4f" % self_measurements['velocity'][0],
                               "%.4f" % self_measurements['velocity'][1],
                               "%.4f" % self_measurements['speed'],
                               "%.4f" % length(arr['forces']['attractive']),
                               "%.4f" % length(arr['forces']['pedestrian_repulsive']),
                               "%.4f" % arr['forces']['repulsive_force'],
                               "%.4f" % arr['forces']['pushing_force'],
                               "%.4f" % arr['forces']['local_density'],
                               "%.4f" % arr['forces']['local_velocity_variance'],
                               "%.4f" % arr['forces']['force_angle'],
                               "%.4f" % arr['forces']['velocity_angle']
                        ]
                        csv_writer.writerow(row)
                    # Empty all data.
                    pedestrian.measurements = []

        # Remove pedestrians from local variables.
        self.all_pedestrians = current_pedestrians

    def build(self):
        """Run the simulation."""
        angle_ou = (0.0, 0.05, 0.005)
        velocity_ou = (0.0, 0.05, 0.04)

        if not os.path.exists(self.storage_folder):
            if self.debug:
                print("Creating directory {}".format(self.storage_folder))
            os.makedirs(self.storage_folder)
        
        if self.debug:
            print("Starting new simulation.")
            print("")

        if self.debug:
            print("Writing pedestrians to {}".format(self.pedestrian_file))
            print("Writing measurements to {}".format(self.measurement_file))

        if not os.path.exists(self.image_target):
            os.makedirs(self.image_target)

        # Build the simulation
        with open(self.parameter_file) as file:
            yaml_data = load(file, Loader=Loader)

        if self.spawn_mode == 3 and self.max_pedestrians:
            for g in range(len(yaml_data['groups'])):
                yaml_data['groups'][g]['num_pedestrians'] = int(self.max_pedestrians / len(yaml_data['groups']))

        if self.labda:
            yaml_data['turbulence_lambda'] = self.labda

        loader = sfm.ParameterLoader(data=yaml_data)
        self.world = loader.world
        starting_time = 0.0

        if self.pedestrian_load_file != '':
            with open(self.pedestrian_load_file) as infile:
                import pickle
                data = pickle.load(infile)
                for p in data:
                    ped = sfm.Pedestrian(group=self.world.groups[0],
                                         radius=p['radius'],
                                         mass=p['mass'],
                                         desired_velocity=p['desired_velocity'],
                                         maximum_velocity=p['maximum_velocity'],
                                         relaxation_time=p['relaxation_time'],
                                         target_path=p['target_path'],
                                         start=p['position'])
                    ped.velocity = p['velocity']
                    ped.next_velocity = p['velocity']
                    ped.speed = p['speed']
                    ped.next_speed = p['speed']
                    self.world.groups[0].add_pedestrian(ped)

            print("Imported {} pedestrians".format(len(self.world.groups[0].get_pedestrians())))

        if os.path.exists(self.pedestrian_file) and not self.resume and not self.force:
            print("Output file {} already exists. Either delete the file, or add the -R (resume) or -F (force) flag.".format(self.pedestrian_file))
            return False

        # If resume flag is enabled, load the data from file.
        # Important: not all stats are restored (such as angle variance)
        # causing measurements to show outliers.
        if self.resume:

            self.world.clear()

            pedestrians = {}
            with open(self.pedestrian_file) as infile:
                reader = csv.reader(infile)
                header = next(reader)
                for row in reader:
                    pedestrians[row[0]] = row
                    pedestrians[row[0]].append(None)

            with open(self.measurement_file) as infile:
                reader = csv.reader(infile)
                header = next(reader)

                for row in reader:
                    t = float(row[0])
                    if t > starting_time:
                        starting_time = t
                    pedestrians[row[1]][-1] = row

            for id_ in pedestrians:
                pedestrian = pedestrians[id_]
                last = pedestrian[-1]
                if last is None:
                    continue
                group_id = int(pedestrian[-2])
                ped = self.world.groups[group_id].spawn_pedestrian()

                ped.position = np.array([float(last[2]), float(last[3])])
                ped.velocity = np.array([float(last[4]), float(last[5])])
                ped.speed = float(last[6])
                ped.next_velocity = ped.velocity
                ped.next_position = ped.position
                ped.next_speed = ped.speed

                ped.mass = float(pedestrian[1])
                ped.radius = float(pedestrian[2])
                ped.desired_velocity = float(pedestrian[3])
                ped.maximum_velocity = float(pedestrian[4])
                ped.id = id_
                self.processed_pedestrians.add(id_)

            self.world.time = starting_time + self.world.step_size
        else:
            with open(self.pedestrian_file, "w") as ped_outfile:
                ped_writer = csv.writer(ped_outfile)
                ped_writer.writerow(['p', 'mass', 'radius', 'desired_velocity', 'maximum_velocity',
                                     'group'])

            with open(self.measurement_file, "w") as csv_outfile:
                csv_writer = csv.writer(csv_outfile)
                csv_writer.writerow(['t', 'p', 'pos_x', 'pos_y', 'vel_x', 'vel_y', 'speed', 
                                     'attractive', 'ped_repulsive', 'repulsive', 'pushing',
                                     'local_density', 'local_velocity_variance', 'force_angle',
                                     'velocity_angle'])

        self.world.update()

        # Set angular variance
        for group in self.world.groups:
            group.set_ornstein_uhlenbeck_process(*angle_ou, process='angle')

        # Spawn pedestrians in a diamond-like structure.
        if self.spawn_mode == 1:

            if not self.max_pedestrians:
                print("Please set -m")
                exit()

            spawn_area = self.world.groups[0].spawn_area
            min_x = spawn_area.start[0]
            max_x = spawn_area.end[0]
            diff_x = max_x - min_x
            min_y = spawn_area.start[1]
            max_y = spawn_area.end[1]
            diff_y = max_y - min_y

            startpos = np.array([min_x, min_y])
            third = np.pi / 3

            difference = 5 * self.world.groups[0].default_radius
            coordinates = []
            translation = [2, 1, -1]

            while len(coordinates) < self.max_pedestrians and difference > 0.15:

                q = Queue()
                q.put(startpos)

                coordinates = []
                coordinates.append(startpos)

                cx, cy = min_x, min_y
                i = 0
                while True:
                    radial = (1 + (i%2)) * third
                    position = np.array([cx, cy]) + difference * np.array([np.cos(radial), np.sin(radial)])
                    if not min_x <= position[0] <= max_x or not min_y <= position[1] <= max_y:
                        break

                    q.put(position)
                    coordinates.append(position)
                    cx, cy = position
                    i += 1

                while len(coordinates) < self.max_pedestrians and not q.empty():
                    cx, cy = q.get()
                    position = np.array([cx + difference, cy])

                    if not min_x <= position[0] <= max_x or not min_y <= position[1] <= max_y:
                        continue

                    q.put(position)
                    coordinates.append(position)

                difference -= 0.01

            # # Add pedestrians to simulation
            group = self.world.groups[0]
            for i, position in enumerate(coordinates):
                p = self.world.groups[0].spawn_pedestrian()
                p.position = position
                p.initialize()
                p.quad.remove(p)
                self.world.quadtree.add(p)

        # Spawn pedestrians randomly at start time.
        elif self.spawn_mode == 2:
            for _ in range(self.max_pedestrians):
                self.world.groups[0].spawn_pedestrian()

        return True

    def run(self):
        bar = progressbar.ProgressBar()
        starting_step = int(200*self.world.time) / int(200*self.world.step_size)
        for step in bar(range(self.steps)):

            if self.debug:
                print("Step {}".format(starting_step + step))

            if not self.world.step():
                break

            # Add newly added pedestrians to our current collection of pedestrians
            for group in self.world.groups:
                for p in group.get_pedestrians():
                    self.all_pedestrians.add(p)

            # Print an image to file if the flag is set
            if self.image and step >= 0 and step % self.image == 0:

                figure = self.world.plot()
                figure.savefig(os.path.join(self.image_target, str(int((starting_step + step + 1) // self.image)).zfill(4) + ".png"),
                               bbox_inches = 'tight',
                               pad_inches = 0.1)
                figure.clear()
                plt.close(figure)

            # Cleanup to avoid high memory usage.
            if step % 50 == 49:
                self.write_intermediary_results()

        self.write_intermediary_results()


def plot(item, measurements, fig, subplot=111):
    """Plot something?"""
    ax = fig.add_subplot(subplot)
    ax.scatter(range(len(measurements)), measurements)
    ax.set_title('average ' + item[2])


def length(x):
    """Calculate the length of a vector."""
    if len(x) == 1:
        return x
    return math.sqrt(x[0]**2 + x[1]**2)

def main(args):
    exp = Experiment(args)
    if exp.build():
        exp.run()

if __name__ == '__main__':
    import argparse
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='YAML-file')
    parser.add_argument('-s', '--steps', help='Number of steps', type=int, default=500)
    parser.add_argument('-d', '--debug', help='Whether to debug the simulation', action='store_true')
    parser.add_argument('-i', '--image', help='Create an image every x steps', type=int, default=0)
    parser.add_argument('-o', '--outfile', help='File for measurements', default='measurements')
    parser.add_argument('-p', '--pedestrian_file', help='Pickle file containing pedestrian positions', default='')
    parser.add_argument('-m', '--max_pedestrians', help='max pedestrians', type=int)
    parser.add_argument('-r', '--repetitions', default=1, type=int)
    parser.add_argument('-R', '--resume', help='Resume a simulation', action='store_true')
    parser.add_argument('-F', '--force', help='Overwrite previous results for simulation', action='store_true')
    parser.add_argument('-x', '--spawn_mode', default=3, type=int)
    parser.add_argument('-l', '--labda', default=0.25, type=float)
    parser.add_argument('-S', '--storage', help='Storage folder', default='results')
    args = parser.parse_args(sys.argv[1:])
    main(args)
