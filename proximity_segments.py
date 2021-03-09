import numpy as np
from tqdm import tqdm


def distance_point2segment(point, line):
    """
    Returns the distance a point and line by computing the distance
    between the point and the point projected to the segment
    :param point: array([x, y])
    :param line: array ([point1.x, point1.y, point2.x, point2.y, width])
    :return: float
    """
    x = point
    u = line[:2]
    v = line[2:4]

    h = np.dot(v - u, x - u) / np.linalg.norm(v - u)**2
    P = u + h * (v - u)
    d = np.linalg.norm(point - P)
    abs_min = min(u[0], v[0])
    abs_max = max(u[0], v[0])
    ord_min = min(u[1], v[1])
    ord_max = max(u[1], v[1])
    if abs_min < x[0] < abs_max and ord_min < x[1] < ord_max:
        return d
    else:
        d_ = min(np.linalg.norm(u-P), np.linalg.norm(v-P))
        return np.hypot(d, d_)


def distance_segment2segment(line_1, line_2):
    """
    Returns the distance between two segments
    :param line_1: array ([point1.x, point1.y, point2.x, point2.y, width])
    :param line_2: array ([point1.x, point1.y, point2.x, point2.y, width])
    :return: float
    """
    d1 = distance_point2segment(line_1[:2], line_2)
    d2 = distance_point2segment(line_1[2:4], line_2)
    d3 = distance_point2segment(line_2[:2], line_1)
    d4 = distance_point2segment(line_2[2:4], line_1)
    return min(d1, d2, d3, d4)


def distance_point2line(point, line):
    """
    Returns the distance a point and line by computing the distance
    between the point and the point projected to the line
    :param point: array([x, y])
    :param line: array ([point1.x, point1.y, point2.x, point2.y, width])
    :return: float
    """
    x = point
    u = line[:2]
    v = line[2:4]

    h = np.dot(v - u, x - u) / np.linalg.norm(v - u)**2
    P = u + h * (v - u)
    d = np.linalg.norm(point - P)
    return d


def distance_line2line(line_1, line_2):
    """
    Returns the distance between two parallel lines
    :param line_1: array ([point1.x, point1.y, point2.x, point2.y, width])
    :param line_2: array ([point1.x, point1.y, point2.x, point2.y, width])
    :return: float
    """
    d = distance_point2segment(line_1[:2], line_2)
    return d


def are_closest_line(index_line_1, index_line_2, index_lines, distance_segment):
    """
    Returns True if there is no segments between line_1 and line_2, will return False if this is not the case
    :param index_line_1:
    :param index_line_2:
    :param index_lines:
    :param distance_segment:
    :return: boolean
    """
    distance_segment_1 = distance_segment[index_line_1, index_lines]
    distance_segment_2 = distance_segment[index_line_2, index_lines]
    distance_1_2 = distance_segment_1[index_lines == index_line_2]
    for index_line in range(len(index_lines)):
        if distance_1_2 > distance_segment_1[index_line] and distance_1_2 > distance_segment_2[index_line]:
            return False
    return True


def return_proximity_segments(lines, index_segments):
    """
    Returns the pairs of index of the segments from index_segments such that
    there is no other aligned and parallel segments between them
    :param lines:
    :param index_segments:
    :return: list of lists of two elements (indexes)
    """
    nb_lines = lines.shape[0]
    nb_matches = len(index_segments)
    reduce_lines = []
    distances = np.zeros((nb_lines, nb_lines))
    index_matching = []
    # compute distances between lines
    for i in tqdm(range(nb_lines)):
        for j in range(i+1, nb_lines):
            distances[i, j] = distance_line2line(lines[i], lines[j])
            distances[j, i] = distances[i, j]
    # compute matching index
    for i in tqdm(range(nb_lines)):
        L = []
        for j in range(nb_matches):
            if index_segments[j][0] == i:
                L.append(index_segments[j][1])
            elif index_segments[j][1] == i:
                L.append(index_segments[j][0])
        index_matching.append(L)
    for (index_1, index_2) in tqdm(index_segments):
        L = index_matching[index_1] + index_matching[index_2]
        if are_closest_line(index_1, index_2, L, distances):
            reduce_lines.append([index_1, index_2])

    return reduce_lines
