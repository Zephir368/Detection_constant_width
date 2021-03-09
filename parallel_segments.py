import numpy as np
from tqdm import tqdm


def angle_of_vectors(a_x, a_y, b_x, b_y):
    """

    :param a_x:
    :param a_y:
    :param b_x:
    :param b_y:
    :return:
    """
    dotProduct = a_x * b_x + a_y * b_y

    modOfVector1 = np.sqrt(a_x * a_x + a_y * a_y) * np.sqrt(b_x * b_x + b_y * b_y)

    angle = dotProduct / modOfVector1
    angleInDegree = np.rad2deg(np.arccos(angle))
    angleInDegree = np.minimum(angleInDegree, 180-angleInDegree)
    return angleInDegree


def compute_parallel_segments(lines, tau):
    """

    :param lines:
    :param tau:
    :return:
    """
    matching_line = []
    nb_lines = lines.shape[0]
    a_y = lines[:, 1]
    a_x = lines[:, 0]
    b_y = lines[:, 3]
    b_x = lines[:, 2]
    vec_x = b_x - a_x
    vec_y = b_y - a_y
    for index_line in tqdm(range(nb_lines - 1)):
        theta = angle_of_vectors(vec_x[index_line], vec_y[index_line], vec_x[index_line + 1:], vec_y[index_line + 1:])
        index_lines = np.arange(index_line + 1, nb_lines)
        indexes_matching = [i for i in index_lines[theta < tau]]
        for index_matching in indexes_matching:
            matching_line.append([index_line, index_matching])
    return matching_line
