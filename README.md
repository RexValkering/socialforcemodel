This package contains a Python implementation of the Social Force Model (Helbing, 1995). The crowd dynamics code is inspired by a C implementation of PedSim (https://github.com/srl-freiburg/pedsim_ros).

The implementation has not yet been fully tested, so use with care.

### Installation

Just checkout the repository and run `python setup.py`.

### TODO

- Confirm implementation correctness
- Allow parameter tuning, such as preferred distance between pedestrians
- Add unit tests
- Add density map to `world.plot()`
- Fix obstacle displaying in `world.plot()`
- Create wireless sensor implementation

### Example code

The following code creates two areas connected by a narrow tunnel where 80
pedestrians try to go from one area to the next.

```python
import socialforcemodel as sfm
import time

# Set parameters.
num_pedestrians = 80
steps = 250

# Create a world.
world = sfm.World(30, 10, step_size = 0.05, desired_velocity=1.5 / 2, maximum_velocity=3.0 / 2)

# Add the first group.
group = sfm.Group(0)
group.set_spawn_area(sfm.Area([1, 1], [9,  9]))
group.set_target_area(sfm.Area([21, 1], [29,  9]))

# Add nodes to the target path, so the pedestrians know which path to follow.
group.add_path_node([9,  5])
group.add_path_node([21,  5])

# Create pedestrians.
for i in range(num_pedestrians):
    group.generate_pedestrian(i)
world.add_group(group)

# Create six walls. Should also work with 2 rectangles.
wall = sfm.Obstacle([[10, 0], [10, 4]])
world.add_obstacle(wall)

wall = sfm.Obstacle([[10, 6], [10, 10]])
world.add_obstacle(wall)

wall = sfm.Obstacle([[10, 4], [20, 4]])
world.add_obstacle(wall)

wall = sfm.Obstacle([[10, 6], [20, 6]])
world.add_obstacle(wall)

wall = sfm.Obstacle([[20, 0], [20, 4]])
world.add_obstacle(wall)

wall = sfm.Obstacle([[20, 6], [20, 10]])
world.add_obstacle(wall)

# Update and plot.
group.update()
figure = world.plot()
figure.savefig("img/0.png")

for step in range(steps):
    print
    print "### STEP {} ###".format(step)
    if not world.step():
        break
    if step % 5 == 4:
        figure = world.plot()
        figure.savefig("img/" + str(step + 1) + ".png")
```

The following code creates a box with 80 pedestrians that keep wandering to
random targets.

```python
import socialforcemodel as sfm
import os
import matplotlib.pyplot as plt

# Simulation parameters.
num_pedestrians = 80
steps = 1000

# Create a world of 11x11 meters.
world = sfm.World(11, 11, step_size = 0.05, desired_velocity=0.75,
                  maximum_velocity=1.5)

# Create a box and add it to the world.
world.add_obstacle(sfm.Obstacle([[0.5, 0.5], [0.5, 10.5]]))
world.add_obstacle(sfm.Obstacle([[0.5, 10.5], [10.5, 10.5]]))
world.add_obstacle(sfm.Obstacle([[0.5, 0.5], [10.5, 10.5]]))
world.add_obstacle(sfm.Obstacle([[10.5, 0.5], [10.5, 10.5]]))

# Add a group and set the spawn and target area. Both the spawn and target
# area are enclosed within the rectangle.
group = sfm.Group(0)
group.set_spawn_area(sfm.Area([1, 1], [10, 10]))
group.set_target_area(sfm.Area([1, 1], [10, 10]))
world.add_group(group)

# This assigns a new target to each pedestrian each time they reach their
# current target.
group.set_final_behaviour('wander')

# Generate the pedestrians.
for i in range(num_pedestrians):
    group.generate_pedestrian(i)
world.add_group(group)

# Generate a starting figure.
if not os.path.exists("img"):
    os.path.makedirs("img")
group.update()
figure = world.plot()
figure.savefig("img/0.png")
plt.close()

for step in range(steps):
    if not world.step():
        break
    # Create an image every 5 steps.
    if step % 5 == 4:
        figure = world.plot()
        figure.savefig("img/" + str(step + 1) + ".png")
        plt.close()
```