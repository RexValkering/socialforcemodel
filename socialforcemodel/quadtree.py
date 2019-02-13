import matplotlib.patches as patches
from .pedestrian import Pedestrian


class QuadTree(object):
    """ Quad Tree implementation that holds world pedestrians.

    A Quad Tree is a method of storing objects with real-world coordinates.
    The tree provides a computationally faster method of retrieving a
    pedestrian's potential neighbours. The root node is initialized with an
    xmin, ymin and length, but is able to increase its area dynamically.

    Args:
        xmin: the tree's starting x
        xmax: the tree's starting y
        length: the initial side length
        threshold: the maximum number of pedestrians in a tree node
    """

    SINGLE = 1
    DIVIDED = 2

    def __init__(self, xmin, ymin, length, threshold):
        # Setting current dimensions
        self.xmin = xmin
        self.ymin = ymin
        self.length = length

        # Setting initial tree variables.
        self.threshold = threshold
        self.type = QuadTree.SINGLE

        self.children = []
        self.pedestrians = set()
        self.count = 0

    def inside(self, pedestrian):
        """ Returns if a pedestrian is inside this Quad Tree node. """
        position = pedestrian
        if isinstance(pedestrian, Pedestrian):
            position = pedestrian.position

        if (position[0] < self.xmin or position[0] > self.xmin + self.length or
                position[1] < self.ymin or position[1] > self.ymin +
                self.length):
            return False
        return True

    def get_subtree_index(self, pedestrian):
        """ Returns the index of the subtree this pedestrian is in. """
        x_index = 0
        y_index = 0
        if pedestrian.position[0] >= self.xmin + 0.5 * self.length:
            x_index = 1
        if pedestrian.position[1] >= self.ymin + 0.5 * self.length:
            y_index = 1
        return x_index + 2 * y_index

    def add(self, pedestrian):
        """ Adds a pedestrian to this tree. """
        if hasattr(pedestrian, 'quad') and pedestrian.quad is not None:
            return

        if pedestrian in self.pedestrians:
            print("Pedestrian already in this set.")
            print(self.count)
            print(self.xmin, self.ymin, self.length)
            exit()
        if self.type == QuadTree.SINGLE:

            # Check if we should divide this QuadTree.
            if self.count == self.threshold:

                # Divide this tree and let the next if-statement handle it.
                self.divide()
            else:
                self.pedestrians.add(pedestrian)
                pedestrian.quad = self

        # If the current tree is divided, add it to the relevant node.
        if self.type == QuadTree.DIVIDED:

            # Add the pedestrian to the subtree.
            index = self.get_subtree_index(pedestrian)
            self.children[index].add(pedestrian)

        # Increase the pedestrian counter.
        self.count += 1

    def remove(self, pedestrian):
        """ Removes a pedestrian from this tree. """

        position = pedestrian.position

        if self.type == QuadTree.DIVIDED:

            # First check if we should merge this QuadTree.
            if self.count == self.threshold:
                # Merge this tree and let the next if-statement handle it.
                self.merge()
            else:
                # Remove the pedestrian from a subtree.
                index = self.get_subtree_index(pedestrian)
                self.children[index].remove(pedestrian)

        if self.type == QuadTree.SINGLE:
            try:
                self.pedestrians.remove(pedestrian)
                pedestrian.quad = None
            except:
                print(pedestrian.quad.type)
                print(pedestrian.quad.xmin, pedestrian.quad.ymin, pedestrian.quad.length)
                print(pedestrian.position)
                print(self.xmin, self.ymin, self.length)
                raise

        # Decrease the pedestrian counter.
        self.count -= 1

    def divide(self):
        """ Divides itself into four subtrees. """
        # Create four subtrees.
        for i in range(4):
            xmin = self.xmin + (i % 2) * 0.5 * self.length
            ymin = self.ymin + int(i / 2) * 0.5 * self.length
            self.children.append(QuadTree(xmin, ymin, 0.5 * self.length,
                                          self.threshold))

        # Move all pedestrians.
        for pedestrian in self.pedestrians:
            index = self.get_subtree_index(pedestrian)
            pedestrian.quad = None
            self.children[index].add(pedestrian)

        # Empty pedestrian set.
        self.pedestrians = set()
        self.type = QuadTree.DIVIDED

    def merge(self):
        """ Merges its four subtrees into one tree. """

        for i in range(4):
            # First merge the subtree if needed.
            if self.children[i].type == QuadTree.DIVIDED:
                self.children[i].merge()

            # Now move the pedestrians.
            for pedestrian in self.children[i].pedestrians:
                self.pedestrians.add(pedestrian)
                pedestrian.quad = self

        # Remove all subtrees.
        self.children = []
        self.type = QuadTree.SINGLE

    def get_pedestrians_in_range(self, position, sensor_range):
        """ Gets the pedestrians that are at most sensor_range away from
        the given position.

        Args:
            position: np.array of x and y
            sensor_range: maximum distance the pedestrians can be away from
                the position

        Returns: set of pedestrians in range.
        """

        # Skip calculations for empty quads.
        if self.count == 0:
            return set()

        # Determine the point closest to the position
        x_point = self.xmin
        y_point = self.ymin

        # Determine x of closest point.
        if position[0] > self.xmin + self.length:
            x_point += self.length
        elif position[0] > self.xmin:
            x_point = position[0]

        # Determine y of closest point.
        if position[1] > self.ymin + self.length:
            y_point += self.length
        elif position[1] > self.ymin:
            y_point = position[1]

        # Calculate distance
        distance_sq = (position[0] - x_point)**2 + (position[1] - y_point)**2
        if distance_sq <= sensor_range**2:

            # The current quad overlaps. If this is a single quad, return
            # its pedestrians, else return pedestrians of overlapping subquads.
            if self.type == QuadTree.SINGLE:
                return self.pedestrians
            else:
                pedestrians = set()
                for child in self.children:
                    pedestrians = pedestrians.union(
                        child.get_pedestrians_in_range(position,
                                                       sensor_range)
                    )
                return pedestrians

        # There is no overlap
        return set()

    def get_number_of_pedestrians_in_box(self, xmin, xmax, ymin, ymax):
        # No pedestrians in quad.
        # print("Considering: ({},{}) x ({},{})".format(xmin, xmax, ymin, ymax))
        if self.count == 0:
            # print("    No pedestrians in ({},{}) x ({}, {})".format(self.xmin, self.xmin + self.length, self.ymin, self.ymin + self.length))
            return 0

        # Quad lies outside target area.
        if ((self.xmin + self.length) <= xmin or self.xmin >= xmax
                or self.ymin >= ymax or (self.ymin + self.length) <= ymin):
            # print("    Quad lies outside target area ({},{}) x ({}, {})".format(self.xmin, self.xmin + self.length, self.ymin, self.ymin + self.length))
            return 0

        # Quad lies fully within target area.
        if (self.xmin >= xmin and (self.xmin + self.length) <= xmax
                and self.ymin >= ymin and (self.ymin + self.length) <= ymax):
            # print("{}   Quad lies inside target area ({},{}) x ({}, {})".format(self.count, self.xmin, self.xmin + self.length, self.ymin, self.ymin + self.length))
            # for p in self.get_pedestrians():
                # print("    -   {}".format(p.position))
            return self.count

        # Quad is both in and out target area. Divide and conquer.
        count = 0
        if self.type == QuadTree.SINGLE:
            for p in self.pedestrians:
                if (xmin <= p.position[0] <= xmax
                        and ymin <= p.position[1] <= ymax):
                    # print("    -   {}".format(p.position))
                    count += 1
        else:
            for child in self.children:
                count += child.get_number_of_pedestrians_in_box(xmin, xmax, ymin, ymax)
        # print("    Found {} pedestrians".format(count))
        return count

    def draw(self, ax):
        """ Draw the Quad Tree on a matplotlib subplot. """
        patch = patches.Rectangle((self.xmin, self.ymin), self.length,
                                  self.length, fill=False, alpha=0.0,
                                  color='grey')
        ax.add_patch(patch)
        if self.type == QuadTree.DIVIDED:
            for child in self.children:
                child.draw(ax)

    def get_pedestrians(self):
        if self.type == QuadTree.SINGLE:
            return self.pedestrians
        pedestrians = []
        for child in self.children:
            pedestrians.extend(list(child.get_pedestrians()))
        return pedestrians
