import socialforcemodel as sfm
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import psutil
from pympler import asizeof, tracker

np.seterr(all='raise')

try:
    import progressbar
except ImportError, e:
    print "Progressbar package not found. Please run 'pip install progressbar'"
    exit()

def sensor(world, position, sensor_range):
    peds = world.quadtree.get_pedestrians_in_range(position, sensor_range)
    actual_peds = set()
    range_squared = sensor_range**2

    for p in peds:
        if ((p.position[0] - position[0])**2 +
                (p.position[1] - position[1])**2) <= range_squared:
            actual_peds.add(p)

    results = {}
    results['count'] = len(actual_peds)
    if len(actual_peds):
        average_speed = 0.0
        for p in actual_peds:
            average_speed += p.speed
        results['average_speed'] = average_speed / len(actual_peds)
    else:
        results['average_speed'] = 0.0
    return results

def sensor_far(world):
    return sensor(world, [14.0, 5.0], 2.0)

def sensor_near(world):
    return sensor(world, [8.0, 5.0], 2.0)

def plot(item, measurements, fig, subplot=111):
    ax = fig.add_subplot(subplot)
    ax.scatter(range(len(measurements)), measurements)
    ax.set_title('average ' + item[2])

def main(args):

    barrier_start = 50.0
    barrier_points = [[50.0, 1.0], [50.0, 4.0]]
    barrier_time = args.barriertime

    mean, theta, sigma = 0.0, 0.05, 0.005

    measurements = []

    for r in range(args.repetitions):
        barrier_state = 0

        if os.path.exists("hddm/{}_pedestrians_{}.csv".format(args.outfile, r)):
            print "Already done, continue..."
            continue

        with open("hddm/{}_pedestrians_{}.csv".format(args.outfile, r), "w") as ped_outfile:
            ped_writer = csv.writer(ped_outfile)
            ped_writer.writerow(['p', 'mass', 'radius', 'desired_velocity', 'maximum_velocity'])

        with open("hddm/{}_measurements_{}.csv".format(args.outfile, r), "w") as csv_outfile:
            csv_writer = csv.writer(csv_outfile)
            csv_writer.writerow(['t', 'p', 'pos_x', 'pos_y', 'vel_x', 'vel_y', 'speed', 'local_density', 'local_velocity_variance'])

        all_pedestrians = set()

        if not os.path.exists("img"):
            os.makedirs("img")
        if not os.path.exists("img/" + args.outfile):
            os.makedirs("img/" + args.outfile)
        if not os.path.exists("measurements"):
            os.makedirs("measurements")

        measurements.append({
            't': [],
            'count_near': [],
            'count_far': [],
            'speed_near': [],
            'speed_far': []
        })

        loader = sfm.ParameterLoader(args.file)
        world = loader.world

        if args.pedestrian_file != '':
            with open(args.pedestrian_file) as infile:
                import pickle
                data = pickle.load(infile)
                # exit()
                for p in data:
                    ped = sfm.Pedestrian(group=world.groups[0],
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
                    world.groups[0].add_pedestrian(ped)

            print "Imported {} pedestrians".format(len(world.groups[0].get_pedestrians()))

        world.update()
        world.groups[0].spawn_max = args.max_pedestrians
        # world.groups[0].set_ornstein_uhlenbeck_process(self, 0, 0.05, 1.0):

        for group in world.groups:
            group.set_ornstein_uhlenbeck_process(mean, theta, sigma)
            
        bar = progressbar.ProgressBar()
        for step in bar(range(args.steps)):

            if not world.step():
                break

            world.update()

            for group in world.groups:
                for p in group.get_pedestrians():
                    all_pedestrians.add(p)

            # if step % 5 == 0:

            #     figure = world.plot()
            #     figure.savefig("img/" + args.outfile + "/" + str((step + 1) // 5).zfill(4) + ".png",
            #                    bbox_inches = 'tight',
            #                    pad_inches = 0.1)
            #     figure.clear()
            #     plt.close(figure)

            # if step % 5 == 0:

            #     near = sensor_near(world)
            #     far = sensor_far(world)

            #     measurements[r]['t'].append(world.time)
            #     measurements[r]['count_near'].append(near['count'])
            #     measurements[r]['count_far'].append(far['count'])
            #     measurements[r]['speed_near'].append(near['average_speed'])
            #     measurements[r]['speed_far'].append(far['average_speed'])

            #     print len(all_pedestrians)

            # Cleanup to avoid high memory usage.
            if step % 200 == 0:

                # tr.print_diff()
                # process = psutil.Process(os.getpid())
                # print "Before:", process.memory_info().rss
                # print len(all_pedestrians)

                # Get all pedestrians no longer in simulation.
                current_pedestrians = set()
                for group in world.groups:
                    current_pedestrians = current_pedestrians.union(group.get_pedestrians())
                retired_pedestrians = all_pedestrians - current_pedestrians

                # Write all pedestrian data to file.
                with open("hddm/{}_pedestrians_{}.csv".format(args.outfile, r), "a") as ped_outfile:
                    with open("hddm/{}_measurements_{}.csv".format(args.outfile, r), "a") as csv_outfile:
                        ped_writer = csv.writer(ped_outfile)
                        csv_writer = csv.writer(csv_outfile)

                        for p in retired_pedestrians:
                            m = p.measurements
                            row = [p.id, "%.4f" % p.mass, "%.4f" % p.radius,
                                   "%.4f" % p.desired_velocity, "%.4f" % p.maximum_velocity]
                            ped_writer.writerow(row)

                        for p in all_pedestrians:
                            m = p.measurements
                            for arr in m:
                                s = arr['self']
                                row = ["%.2f" % s['time'], p.id, "%.4f" % s['position'][0], "%.4f" % s['position'][1],
                                       "%.4f" % s['velocity'][0], "%.4f" % s['velocity'][1], "%.4f" % s['speed'],
                                       "%.4f" % arr['forces']['local_density'], "%.4f" % arr['forces']['local_velocity_variance']]
                                csv_writer.writerow(row)
                            # Empty all data.
                            p.measurements = []

                # Remove pedestrians from local variables.
                all_pedestrians = current_pedestrians

                # process = psutil.Process(os.getpid())
                # print "After:", process.memory_info().rss

            if barrier_state == 0 and barrier_time != 0 and world.time > barrier_start:
                barrier_state = 1
                world.add_obstacle(sfm.Obstacle(barrier_points))
            elif barrier_state == 1 and world.time > barrier_start + barrier_time:
                barrier_state = 2
                del world.obstacles[-1]

        histogram = None

        # Write all pedestrian data to file.
        with open("hddm/{}_pedestrians_{}.csv".format(args.outfile, r), "a") as ped_outfile:
            with open("hddm/{}_measurements_{}.csv".format(args.outfile, r), "a") as csv_outfile:
                ped_writer = csv.writer(ped_outfile)
                csv_writer = csv.writer(csv_outfile)

                for p in all_pedestrians:
                    if p.id == 0:
                        histogram = [m['self']['random'] for m in p.measurements]

                    m = p.measurements
                    row = [p.id, "%.4f" % p.mass, "%.4f" % p.radius,
                           "%.4f" % p.desired_velocity, "%.4f" % p.maximum_velocity]
                    ped_writer.writerow(row)

                    m = p.measurements
                    for arr in m:
                        s = arr['self']
                        row = ["%.2f" % s['time'], p.id, "%.4f" % s['position'][0], "%.4f" % s['position'][1],
                               "%.4f" % s['velocity'][0], "%.4f" % s['velocity'][1], "%.4f" % s['speed'],
                               "%.4f" % arr['forces']['local_density'], "%.4f" % arr['forces']['local_velocity_variance']]
                        csv_writer.writerow(row)

        # plt.clf()
        # plt.hist(histogram)
        # plt.show()

if __name__ == '__main__':
    import argparse
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='YAML-file')
    parser.add_argument('-s', '--steps', help='Number of steps', type=int, default=500)
    parser.add_argument('-o', '--outfile', help='File for measurements', default='measurements')
    parser.add_argument('-p', '--pedestrian_file', help='Pedestrian file', default='')
    parser.add_argument('-m', '--max_pedestrians', help='max pedestrians', type=int, default=100)
    parser.add_argument('-r', '--repetitions', default=1, type=int)
    parser.add_argument('-b', '--barriertime', default=0, type=int)
    args = parser.parse_args(sys.argv[1:])
    main(args)