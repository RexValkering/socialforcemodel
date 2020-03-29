import os

import socialforcemodel as sfm
import numpy as np
import matplotlib.pyplot as plt

def average_speed(world):
    velocities = []
    for group in world.groups:
        for p in group.pedestrians:
            velocities.append(p.speed)
    return np.mean(velocities)

def main(args):
    # Create image directory if it does not exist.
    image_directory = "img" 
    if not os.path.exists(image_directory):
        os.makedirs(image_directory)

    loader = sfm.ParameterLoader(args.file)
    world = loader.world
    world.update()

    world.add_measurement(average_speed)
    figure = world.plot()
    
    figure.savefig("{}/0.png".format(image_directory),
                   bbox_inches = 'tight',
                   pad_inches = 0.1)
    figure.clear()
    plt.close(figure)

    for step in range(args.steps):
        print("Step {}".format(step + 1))
        if not world.step():
            break
        world.update()
        if step % 5 == 4:
            figure = world.plot()
            figure.savefig("{}/{}.png".format(image_directory, (step + 1) // 5),
                           bbox_inches = 'tight',
                           pad_inches = 0.1)
            figure.clear()
            plt.close(figure)

    np.savetxt("measurements.txt", world.measurements)

if __name__ == '__main__':
    import argparse
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--file', default="simulation.yml", help='YAML-file')
    parser.add_argument('-s', '--steps', help='Number of steps', type=int, default=500)
    args = parser.parse_args(sys.argv[1:])
    main(args)