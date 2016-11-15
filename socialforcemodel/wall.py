from .obstacle import Obstacle


class Wall(Obstacle):
    """ A wrapper for an Obstacle with two points.

    Args:
        start: first x,y point of wall
        end: second x,y point of wall
    """

    def __init__(self, start, end):
        Obstacle.__init__(self, [start, end])
