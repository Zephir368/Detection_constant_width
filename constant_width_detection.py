import numpy as np

from parallel_segments import compute_parallel_segments
from aligned_segments import return_aligned_segment


def distance_middle_segment(line_1, lines):
    """
    Returns the distance between the middle of segment line_1 and a list of segments
    :param line_1: ([point1.x, point1.y, point2.x, point2.y, width])
    :param lines: list of line
    :return:
    """
    middle_1 = return_middle(line_1)
    middles = return_middle(lines.T)
    return np.linalg.norm(middle_1.reshape((2, 1)) - middles, axis=0)


def return_middle(line):
    """
    Returns the coordinates of the middle of a given segment
    :param line: ([point1.x, point1.y, point2.x, point2.y, width])
    :return: float
    """
    return 1 / 2 * np.array([line[0] + line[2], line[1] + line[3]])


def reduce_pairs(lines, index_segments):
    """

    :param lines:
    :param index_segments:
    :return:
    """
    nb_lines = lines.shape[0]
    reduce_lines = []
    distances = np.zeros((nb_lines, nb_lines))
    for i in range(nb_lines):
        distances[i] = distance_middle_segment(lines[i], lines)
    for (index_1, index_2) in index_segments:
        L = [index_segments[i][1] for i in range(len(index_segments)) if index_segments[i][0] == index_1]
        L += [index_segments[i][0] for i in range(len(index_segments)) if index_segments[i][1] == index_1]
        L += [index_segments[i][1] for i in range(len(index_segments)) if index_segments[i][0] == index_2]
        L += [index_segments[i][0] for i in range(len(index_segments)) if index_segments[i][1] == index_2]
        L = np.array(L)
        #L = remove
        index_matched_line_1 = L[np.argmin(distances[index_1, L])]
        index_matched_line_2 = L[np.argmin(distances[index_2, L])]
        if index_matched_line_1 == index_2 and index_matched_line_2 == index_1:
            reduce_lines.append([index_1, index_2])

    return reduce_lines


def constant_width(img, lines, tau=20, threshold=.5):
    """

    :param img:
    :param lines:
    :param tau:
    :return:
    """
    # test parallel segments
    index_segments = compute_parallel_segments(lines, tau)
    print(index_segments)
    # test aligned segments
    index_segments = return_aligned_segment(lines, index_segments, threshold=threshold)
    print(index_segments)
    # test gradient
    #
    # test proximity
    #index_segments = reduce_pairs(lines, index_segments)
    print(index_segments)
    return index_segments
