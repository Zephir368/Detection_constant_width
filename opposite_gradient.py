import numpy as np


def compute_gradient(img):
    # computes the gradient of the image 'im'
    # image size
    nr, nc = img.shape

    gx = img[:, 1:] - img[:, 0:-1]
    gx = np.block([gx, np.zeros((nr, 1))])

    gy = img[1:, :] - img[0:-1, :]
    gy = np.block([[gy], [np.zeros((1, nc))]])
    return gx, gy


def return_opposed_gradient(img, lines, index_parallel_segments):
    gradient_img = compute_gradient(img)
    lines = lines.round().astype(np.int)
    for (index_1, index_2) in index_parallel_segments:
        line_1 = lines[index_1]
        line_2 = lines[index_2]
        a_x, b_x = min(line_1[0], line_1[2]), max(line_1[0], line_1[2]) + 1
        a_y, b_y = min(line_1[1], line_1[3]), max(line_1[1], line_1[3]) + 1
        grad_line_1 = gradient_img[a_x:b_x, a_y:b_y]
        a_x, b_x = min(line_2[0], line_2[2]), max(line_2[0], line_2[2]) + 1
        a_y, b_y = min(line_2[1], line_2[3]), max(line_2[1], line_2[3]) + 1
        grad_line_2 = gradient_img[a_x:b_x, a_y:b_y]
