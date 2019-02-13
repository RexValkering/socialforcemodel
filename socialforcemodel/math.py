import numpy as np

def length_squared(vector):
    """ Return the length squared of a vector. """
    return vector[0]**2 + vector[1]**2

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.sqrt(length_squared(vector))

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    try:
        v1_u = unit_vector(v1)
        v2_u = unit_vector(v2)
    except:
        return 0.0

    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))