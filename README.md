This package contains a Python implementation of the Helbing-Molnár-Farkas-Vicsek Social Force Model. The code was created for a master thesis and is still in active development.

Feel free to clone or fork. For questions regarding the model you can contact me at *[myfullname] at gmail dot com*.

### Installation

Just checkout the repository and run `python setup.py`.

### TODO

- Add optimizations to social force calculations.
- Add GUI to allow run-time parameter tuning and testing.
- Add unit tests.
- Let parameterloader give errors for unknown parameters.
- Make parameterloader case insensitive.

### YAML config files

You can use a .yaml parameter file to load and build a world. The following parameters are configurable:

##### Global parameters

* `world_width` (*float*) - default `10.0` - width (x) of domain
* `world_height` (*float*) - default `10.0` - height (y) of domain
* `continuous_domain` (*boolean*) - default `False` - whether the domain should wrap around
* `step_size` (*float*) - default `0.05` - simulation step size
* `default_desired_velocity` (*float*) - default `1.3` - default desired velocity of pedestrians
* `default_maximum_velocity` (*float*) - default `2.6` - default maximum velocity of pedestrians
* `default_relaxation_time` (*float*) - default `2.0` - default relaxation time of pedestrians
* `desired_velocity_importance` (*float*) - default `0.8` - between 0.0 and 1.0, lower means the velocity is more dependent on neighbourhood velocity
* `interactive_distance_threshold` (*float*) - default `2.0` - distance after which objects and pedestrians are no longer used in interactive force calculations
* `target_distance_threshold` (*float*) - default `0.13` - maximum distance to target for it to be considered reached
* `repulsion_coefficient` (*float*) - default `2000 Newton`
* `falloff_length` (*float*) - default `0.08 meters`
* `body_force_constant` (*float*) - default `12000 kg / s²`
* `friction_force_constant` (*float*) - default `24000 kg / ms`
* `quad_tree_threshold` (*integer*) - maximum number of pedestrians in a quad tree leaf
* `groups` (one or more *Group* entities) - pedestrian groups in this simulation
* `obstacles` (one or more *Obstacle* entities) - obstacles in this simulation

##### Entity variables

###### Point:
* `x` (*float*)
* `y` (*float*)

###### Area:
* `start` (*Point*)
* `end` (*Point*)
    
###### Group:
Default values are taken from global variables if not provided.
* `start_time` (*float*) - default `0 seconds` - Time at which this group should appear in the simulation
* `spawn_area` (*Area*) - The default area in which new pedestrians should spawn
* `spawn_rate` (*float*) - default `0 pedestrians per second` - Pedestrian spawn rate
* `target_area` (*Area*) - The default area in which new pedestrian targets should spawn
* `target_path` (one or more *Point* entities) - The path which pedestrians should follow to get to their target
* `mass` (*float*) - default `60 kg` - Mass of pedestrians
* `radius` (*float*) - default `0.15 m` - Radius of pedestrians
* `desired_velocity` (*float*) - Desired velocity of pedestrians
* `maximum_velocity` (*float*) - Maximum velocity of pedestrians
* `relaxation_time` (*float*) - Relaxation time of pedestrians
* `num_pedestrians` (*integer*) - The number of pedestrians to spawn with default parameters
* `pedestrians` (one or more *Pedestrian* entities) - Pedestrians to add to this group

###### Pedestrian:
Pedestrians should always be part of a group. Variables that are not set are inferred from the group.
* `start` (*Point*) - Spawn point
* `target` (*Point*) - Target point
* `target_path` (one or more *Point* entities) - The path which this pedestrian should follow to get to their target.
* `mass` (*float*) - Mass of pedestrian
* `radius` (*float*) - Radius of pedestrian
* `desired_velocity` (*float*) - Desired velocity of pedestrian
* `maximum_velocity` (*float*) - Maximum velocity of pedestrian
* `relaxation_time` (*float*) - Relaxation time of pedestrian

###### Obstacle:
* `points` (one or more *Point* entities) - series of points. Line segments are drawn between pairs of points.

### Example code

#### Box with exit tunnel
The following parameter file creates a 10x10 box connected to a 10x4 exit tunnel. The box starts with 100 pedestrians spawned at random positions in the box. Each pedestrian then attempts to navigate through the tunnel using a predefined target path, towards an off-screen target beyond the tunnel.

```yaml
world_height: 10
world_width: 20
groups:
    -   num_pedestrians: 100
        spawn_area:
            start:
                x: 0.5
                y: 0.5
            end:
                x: 9.5
                y: 9.5
        target_area:
            start:
                x: 20.0
                y: 2.5
            end:
                x: 25.0
                y: 7.5
        target_path:
            -   x: 9.5
                y: 5.0
            -   x: 21.0
                y: 5.0
obstacles:
    -   points:
            -   x: 10.0
                y: 0.0
            -   x: 10.0
                y: 2.5
            -   x: 11.0
                y: 3.5
            -   x: 20.0
                y: 3.5
    -   points:
            -   x: 20.0
                y: 6.5
            -   x: 11.0
                y: 6.5
            -   x: 10.0
                y: 7.5
            -   x: 10.0
                y: 10.0
```

The following Python file loads the parameters from the YAML file, creates a simulation, adds some measurements and runs and plots the situation.

```python
import socialforcemodel as sfm
import numpy as np
import matplotlib.pyplot as plt

def average_speed(world):
    velocities = []
    for group in world.groups:
        for p in group.pedestrians:
            velocities.append(p.speed)
    return np.mean(velocities)

def avg_num_neighbours(world):
    counts = []
    for group in world.groups:
        for p in group.pedestrians:
            counts.append(p.get_measurement('neighbourhood', 'num_neighbours'))
    return np.mean(counts)

def main(args):
    loader = sfm.ParameterLoader(args.file)
    world = loader.world
    world.update()

    world.add_measurement(average_speed)
    world.add_measurement(avg_num_neighbours)
    figure = world.plot()
    figure.savefig("img/0.png",
                   bbox_inches = 'tight',
                   pad_inches = 0.1)
    figure.clear()
    plt.close(figure)

    for step in range(args.steps):
        print "Step {}".format(step + 1)
        if not world.step():
            break
        world.update()
        if step % 5 == 4:
            figure = world.plot()
            figure.savefig("img/" + str((step + 1) / 5) + ".png",
                           bbox_inches = 'tight',
                           pad_inches = 0.1)
            figure.clear()
            plt.close(figure)

    np.savetxt("measurements.txt", world.measurements)

if __name__ == '__main__':
    import argparse
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='YAML-file')
    parser.add_argument('-s', '--steps', help='Number of steps', type=int, default=500)
    args = parser.parse_args(sys.argv[1:])
    main(args)
```
