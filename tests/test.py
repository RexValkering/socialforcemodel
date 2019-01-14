import socialforcemodel as sfm
import numpy as np
import matplotlib.pyplot as plt
import os

try:
    import progressbar
except ImportError as e:
    print("Progressbar package not found. Please run 'pip install progressbar'")
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

    mean, theta, sigma = 1.3, 0.15, 0.01

    measurements = []

    for r in range(args.repetitions):

        if not os.path.exists("img"):
            os.makedirs("img")
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
        world.update()
        barrier_state = 0

        for group in world.groups:
            group.set_ornstein_uhlenbeck_process(mean, theta, sigma)

        bar = progressbar.ProgressBar()
        for step in bar(range(args.steps)):

            if not world.step():
                break

            world.update()
            if step % 5 == 0:
                figure = world.plot()
                figure.savefig("img/%03d.png" % ((step + 1) // 5),
                               bbox_inches = 'tight',
                               pad_inches = 0.1)
                figure.clear()
                plt.close(figure)

            if step % 5 == 0:

                near = sensor_near(world)
                far = sensor_far(world)

                measurements[r]['t'].append(world.time)
                measurements[r]['count_near'].append(near['count'])
                measurements[r]['count_far'].append(far['count'])
                measurements[r]['speed_near'].append(near['average_speed'])
                measurements[r]['speed_far'].append(far['average_speed'])

    types = ['count_near', 'count_far', 'speed_near', 'speed_far']
    m = measurements

    for i in range(4):
        with open("measurements/{}_{}.csv".format(args.outfile, types[i]), "w") as outfile:
            import csv
            writer = csv.writer(outfile)
            writer.writerow(['t'] + [str(r) for r in range(args.repetitions)])

            for j in range(len(m[r]['t'])):
                row = [m[r]['t'][j]] + [m[r][types[i]][j] for r in range(args.repetitions)]
                writer.writerow(row)

if __name__ == '__main__':
    import argparse
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='YAML-file')
    parser.add_argument('-s', '--steps', help='Number of steps', type=int, default=500)
    parser.add_argument('-o', '--outfile', help='File for measurements', default='measurements')
    parser.add_argument('-r', '--repetitions', default=1, type=int)
    args = parser.parse_args(sys.argv[1:])
    main(args)
