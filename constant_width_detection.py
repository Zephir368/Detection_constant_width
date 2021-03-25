import numpy as np
from pylsd.lsd import lsd

from parallel_segments import compute_parallel_segments
from aligned_segments import return_aligned_segments
from opposite_gradient import return_opposed_gradient
from proximity_segments import return_proximity_segments

from NFA import return_NFA


def return_middle(line):
    """
    Returns the coordinates of the middle of a given segment
    :param line: ([point1.x, point1.y, point2.x, point2.y, width])
    :return: float
    """
    return 1 / 2 * np.array([line[0] + line[2], line[1] + line[3]])


def constant_width(img, tau=20, thres=.5, compute_param=False):
    """

    :param img:
    :param tau:
    :param thres:
    :param compute_param:
    :return:
    """
    lines = lsd(img)
    if compute_param:
        thres_values = np.array([.1 * n for n in range(10)])
        NFA_values = np.zeros(10)
        for i in range(10):
            NFA_values[i] = return_NFA(img, tau, thres_values[i])
        best_index = np.argmin((NFA_values - 1)**2)
        threshold = thres_values[best_index]
        print("threshold = ", threshold)
    else:
        threshold = thres

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

    print(f"\n****   NFA= {return_NFA(img, tau, threshold)}   *****")

    return index_segments
