import numpy as np


class Obstacle(object):
    """ Obstacle class for the social force model.

    An obstacle can be formed consisting of any positive number of points. This
    class contains functions for finding the closest point and computing
    intersections.

    Args:
        points: A point or list of x,y points that define the obstacle's edges.
    """

    def __init__(self, points):

        if not len(points):
            raise ValueError("Incorrect argument")

        # Check if the user simply supplied one point instead of a list of
        # points. This is okay.
        if len(points) == 2 and not hasattr(points[0], '__iter__'):
            points = [points]

        self.points = np.array(points)
        self.min = np.array(points[0])
        self.max = np.array(points[0])

        # Find correct min and max values.
        for point in points[1:]:
            if point[0] < self.min[0]:
                self.min[0] = point[0]
            if point[0] > self.max[0]:
                self.max[0] = point[0]

            if point[1] < self.min[1]:
                self.min[1] = point[1]
            if point[1] > self.max[1]:
                self.max[1] = point[1]

    def pairs(self):
        """ Returns the pairs of points for this polygon. """
        if len(self.points) == 1:
            return []
        if len(self.points) == 2:
            return [(self.points[0], self.points[1])]
        return [(self.points[i], self.points[(i + 1) % len(self.points)])
                for i in range(len(self.points))]

    def closest_point(self, pos, threshold=None):
        """ Returns the closest point of this obstacle to a position.

        Get the point of this obstacle that is closest to a given position.
        This works for obstacles of any number of points. A distance threshold
        can be set to increase the performance of this function, as obstacles
        that are fully beyond the threshold will not return a closest point,
        but None instead.

        Args:
            pos (np.array): the position for which to find the closest point.
            threshold (float): maximum distance to consider.

        Returns:
            np.array: two-dimensional array of closest point, or None if no
                closest point was found.
            np.array: normal of obstacle pointing towards pos
        """

        # If this obstacle has only one point, return it.
        if len(self.points) == 1:
            point = self.points[0]
            distance = np.linalg.norm(point - pos)

            # If an obstacle distance threshold is provided, check if the
            # distance is below it. If not, return no closest point.
            if threshold is not None and distance > threshold:
                point = None
            return point, (pos - point) / distance

        # If an obstacle distance threshold is provided, check if the distance
        # to the bounding box exceeds this threshold. If so, return no closest
        # point. This may improve performance if many obstacles are present.
        if threshold is not None:
            if pos[0] > self.max[0] or pos[0] < self.min[0]:

                # Calculate the x-difference.
                xdiff = min(abs(self.max[0] - pos[0]),
                            abs(self.min[0] - pos[0]))
                if xdiff > threshold:
                    return None, None

                if pos[1] > self.max[1] or pos[1] < self.min[1]:

                    # Calculate the y-difference:
                    ydiff = min(abs(self.max[1] - pos[1]),
                                abs(self.min[1] - pos[1]))
                    distance = np.sqrt(xdiff**2 + ydiff**2)
                    if distance > threshold:
                        return None, None
            else:
                ydiff = min(abs(self.max[1] - pos[1]),
                            abs(self.min[1] - pos[1]))
                if ydiff > threshold:
                    return None, None

        # Loop through all pairs of points and for each line, find the point
        # that is closest to the provided position.
        smallest_distance = np.inf
        closest_point = None
        normal = None
        closest_pair = None
        last_ratio = None

        for first, second in self.pairs():

            difference = second - first
            length_squared = np.linalg.norm(difference)**2
            ratio = max(0, min(1, np.dot(pos - first, difference) /
                               length_squared))
            projection = first + ratio * difference
            distance = np.linalg.norm(pos - projection)

            if distance < smallest_distance:
                smallest_distance = distance
                closest_point = (1 - ratio) * first + ratio * second
                closest_pair = (first, second)
                last_ratio = ratio

        # Don't return a closest point if the distance is beyond the threshold.
        if threshold is not None and smallest_distance > threshold:
            closest_point = None

        if closest_point is not None:
            if ratio < 0.0001 or ratio > 0.9999:
                normal = (pos - closest_point) / smallest_distance
            else:
                diff_x = closest_pair[1][0] - closest_pair[0][0]
                diff_y = closest_pair[1][1] - closest_pair[0][1]
                normal = np.array([-diff_y, diff_x])
                normal = normal / np.linalg.norm(normal)
                off_point = closest_point + normal * 0.5 * smallest_distance
                if np.linalg.norm(pos - off_point) > smallest_distance:
                    normal = -normal

        return closest_point, normal

    def intersects(self, p1, p2):
        """ Calculate intersection of polygon and line formed by p1 and p2.

        Args:
            p1: first point
            p2: second point

        Returns:
            np.array if intersection, else None
        """

        if (p1[0] < self.min[0] and p2[0] < self.min[0]) or \
                (p1[0] > self.max[0] and p2[0] > self.max[0]) or \
                (p1[1] < self.min[1] and p2[1] < self.min[1]) or \
                (p1[1] > self.max[1] and p2[1] > self.max[1]):
            return None

        # If this is a point and it lies exactly between p1 and p2, there is
        # an intersection.
        if len(self.points) == 1:
            return self.points[0]

        def perp(value):
            result = np.empty_like(value)
            result[0] = -value[1]
            result[1] = value[0]
            return result

        def seg_intersect(first_a, second_a, first_b, second_b):
            diff_a = second_a - first_a
            diff_b = second_b - first_b
            diff_p = first_a - first_b
            dap = perp(diff_a)
            denom = np.dot(dap, diff_b)
            num = np.dot(dap, diff_p)
            return (num / denom.astype(float)) * diff_b + first_b

        smallest_distance = np.inf
        closest_point = None

        for start, end in self.pairs():
            intersection = seg_intersect(start, end, p1, p2)

            if intersection[0] == np.inf or intersection[1] == np.inf:
                continue

            if np.linalg.norm(intersection - p1) < smallest_distance:
                smallest_distance = np.linalg.norm(intersection - p1)
                closest_point = np.array([intersection[0], intersection[1]])

        if closest_point is not None:
            return closest_point
        return None
