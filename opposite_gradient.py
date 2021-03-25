import numpy as np
from tqdm import tqdm


def compute_gradient(img):
    """
    Returns the gradient in x and y of the image img
    :param img: image array 2D
    :return:
    """
    nr, nc = img.shape

    gx = img[:, 1:] - img[:, 0:-1]
    gx = np.block([gx, np.zeros((nr, 1))])

    gy = img[1:, :] - img[0:-1, :]
    gy = np.block([[gy], [np.zeros((1, nc))]])
    return gx, gy


def return_gradient_restriction(gradient_img, line):
    a_y, b_y = int(min(line[0], line[2])), int(max(line[0], line[2])) + 1
    a_x, b_x = int(min(line[1], line[3])), int(max(line[1], line[3])) + 1
    nb_points = max(b_x - a_x, b_y - a_y) + 1
    grad_line_index_x = np.linspace(a_x, b_x, nb_points).round().astype(np.int)
    grad_line_index_y = np.linspace(a_y, b_y, nb_points).round().astype(np.int)
    grad_line_x = np.zeros(nb_points)
    grad_line_y = np.zeros(nb_points)
    for i in range(nb_points):
        grad_line_x[i] = gradient_img[0][grad_line_index_x[i], grad_line_index_y[i]]
        grad_line_y[i] = gradient_img[1][grad_line_index_x[i], grad_line_index_y[i]]
    return grad_line_x, grad_line_y


def return_opposed_gradient_from_img(img, lines, index_segments):
    gradient_img = compute_gradient(img.astype(np.float))
    lines = lines.round().astype(np.int)
    reduce_index = []
    for (index_1, index_2) in tqdm(index_segments):
        line_1 = lines[index_1]
        line_2 = lines[index_2]
        grad_line_x_1, grad_line_y_1 = return_gradient_restriction(gradient_img, line_1)
        grad_line_x_1, grad_line_y_1 = np.mean(np.sign(grad_line_x_1)), np.mean(np.sign(grad_line_y_1))
        grad_line_x_2, grad_line_y_2 = return_gradient_restriction(gradient_img, line_2)
        grad_line_x_2, grad_line_y_2 = np.mean(np.sign(grad_line_x_2)), np.mean(np.sign(grad_line_y_2))
        if grad_line_x_1 * grad_line_x_2 < 0 or grad_line_y_1 * grad_line_y_2 < 0:
            reduce_index.append([index_1, index_2])


def distance2points(pt1, pt2):
    """
    Returns the distance between pt1 and pt2
    :param pt1: array ([point1.x, point1.y])
    :param pt2: array ([point2.x, point2.y])
    :return: float
    """
    return np.linalg.norm(pt1-pt2)


def return_opposed_gradient(lines, index_segments):
    """
    Returns the pairs of index of the segments from index_segments such that
    the restrictions of the gradients of the segments are opposite
    :param lines:
    :param index_segments:
    :return: list of lists of two elements (indexes)
    """
    opposed_gradient_index_segments = []
    for (index_1, index_2) in tqdm(index_segments):
        pt1 = lines[index_1, 0:2]
        pt2 = lines[index_1, 2:4]
        pt3 = lines[index_2, 0:2]
        pt4 = lines[index_2, 2:4]
        if distance2points(pt2, pt3) < distance2points(pt2, pt4):# \
                #and distance2points(pt1, pt4) < distance2points(pt1, pt3):
            opposed_gradient_index_segments.append([index_1, index_2])
    return opposed_gradient_index_segments
