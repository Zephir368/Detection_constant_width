import numpy as np

from parallel_segments import compute_parallel_segments
from aligned_segments import return_aligned_segments
from opposite_gradient import return_opposed_gradient
from proximity_segments import return_proximity_segments


def return_middle(line):
    """
    Returns the coordinates of the middle of a given segment
    :param line: ([point1.x, point1.y, point2.x, point2.y, width])
    :return: float
    """
    return 1 / 2 * np.array([line[0] + line[2], line[1] + line[3]])


def constant_width(lines, tau=20, threshold=.5):
    """

    :param img:
    :param lines:
    :param tau:
    :param threshold:
    :return:
    """
    # test parallel segments
    print("\n****   begin parallel test   *****")
    index_segments = compute_parallel_segments(lines, tau=tau)

    # test gradient
    print("\n****   begin gradient test   *****")
    index_segments = return_opposed_gradient(lines, index_segments)

    # test aligned segments
    print("\n****   begin aligned test   *****")
    index_segments = return_aligned_segments(lines, index_segments, threshold=threshold)

    # test proximity
    print("\n****   begin proximity test   *****")
    index_segments = return_proximity_segments(lines, index_segments)
    return index_segments
