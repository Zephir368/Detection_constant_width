import numpy as np


def length_segment(line):
    """

    :param line:
    :return:
    """
    point_1 = line[:2]
    point_2 = line[2:4]
    return np.linalg.norm(point_1 - point_2)


def project_point_to_line(point, line):
    """
    Returns the coordinates of the point projected to the line
    :param point: array([x, y])
    :param line: array([point1.x, point1.y, point2.x, point2.y, width])
    :return: array([P_x, P_y])
    """
    x = point
    u = line[:2]
    v = line[2:4]

    d = np.dot(v - u, x - u) / np.linalg.norm(v - u)
    P = u + d * (v - u) / np.linalg.norm(v - u)
    return P


def segment_intersection(line_1, line_2):
    """
    Returns the length of the intersection segment of line_1 projected to line_2 and line_2
    :param line_1:
    :param line_2:
    :return: float between 0 and 1
    """
    a_1 = line_1[0:2]
    b_1 = line_1[2:4]
    P_1 = project_point_to_line(a_1, line_2)
    P_2 = project_point_to_line(b_1, line_2)
    projected_line_1 = np.array((P_1[0], P_1[1], P_2[0], P_2[1]))

    left = max(min(projected_line_1[0], projected_line_1[2]), min(line_2[0], line_2[2]))
    right = min(max(projected_line_1[0], projected_line_1[2]), max(line_2[0], line_2[2]))
    top = max(min(projected_line_1[1], projected_line_1[3]), min(line_2[1], line_2[3]))
    bottom = min(max(projected_line_1[1], projected_line_1[3]), max(line_2[1], line_2[3]))
    point_1_inter = np.array((left, bottom))
    point_2_inter = np.array((right, top))
    return np.linalg.norm(point_1_inter - point_2_inter) / max(length_segment(line_1), length_segment(line_2))


def return_aligned_segment(lines, index_segments, threshold=.7):
    aligned_index_segments = []
    for (index_1, index_2) in index_segments:
        alignment_measure = segment_intersection(lines[index_1], lines[index_2])
        if alignment_measure > threshold:
            aligned_index_segments.append([index_1, index_2])
    return aligned_index_segments