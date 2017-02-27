from .obstacle import Obstacle
import numpy as np


class MeasureBarrier(Obstacle):
    """ A barrier that counts the number of pedestrians that pass through it.

    The user can define a barrier by defining two points of the barrier. The
    way the barrier works is that it first requires the pedestrian to enter
    the barrier, then leave it in the same direction as it entered. This is to
    avoid pedestrians being counted multiple times due to stepping back and
    forth.

    Args:
        world: the world in which the barrier is placed
        start: first x,y point of barrier
        end: second x,y point of barrier
        direction: the direction in which pedestrians must pass through the
                   barrier to count. The following options are legal:
                   'up', 'left', 'down' or 'right'

    """

    def __init__(self, world, start, end, direction):
        self.world = world
        self.direction = direction
        self.count = 0
        self.inside = set()

        # Convert start and end to numpy vectors.
        start = np.array(start).astype(float)
        end = np.array(end).astype(float)

        # Determine the width of the barrier.
        barrier_width = world.maximum_velocity * world.step_size

        points = [np.array(start), np.array(end), np.array(end),
                  np.array(start)]

        # Get the normal vector
        diff = end - start
        normal = np.array([-diff[1], diff[0]])
        normal = normal / np.linalg.norm(normal)

        # Check if the direction argument is useful.
        if (((direction == 'up' or direction == 'down') and normal[1] == 0) or
                ((direction == 'left' or direction == 'right') and
                    normal[0] == 0)):
            print "Incorrect direction supplied for barrier."
            return

        # Determine the points.
        if direction == 'up' or direction == 'down':
            # Check if we need to use the reverse normal.
            if ((normal[1] < 0 and direction == 'up') or
                    (normal[1] > 0 and direction == 'down')):
                normal = - normal

            # Subtract the normal to the points.
            points[2] -= barrier_width * normal
            points[3] -= barrier_width * normal
        elif direction == 'left' or direction == 'right':
            # Check if we need to use the reverse normal.
            if ((normal[0] > 0 and direction == 'left') or
                    (normal[0] < 0 and direction == 'right')):
                normal = - normal
            # Subtract the normal to the points.
            points[2] -= barrier_width * normal
            points[3] -= barrier_width * normal

        Obstacle.__init__(self, points)

    def check_if_passed(self, pedestrian):
        pos = pedestrian.position
        next_pos = pedestrian.next_position
        # First check if the pedestrian intersects with a barrier edge.
        if self.intersects(pos, next_pos) is not None:

            # Check if the barrier edge is crossed correctly.
            if ((self.direction == 'right' and next_pos[0] > pos[0]) or
                    (self.direction == 'left' and next_pos[0] < pos[0]) or
                    (self.direction == 'up' and next_pos[1] > pos[1]) or
                    (self.direction == 'down' and next_pos[1] < pos[1])):

                # Only count the pedestrian once it leaves the barrier.
                if pedestrian not in self.inside:
                    self.inside.add(pedestrian)
                else:
                    self.inside.remove(pedestrian)
                    self.count += 1
                    return True

        return False
