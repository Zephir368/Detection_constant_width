import numpy as np
from pylsd.lsd import lsd

from aligned_segments import length_segment
import skimage.io as io


def return_distribution_length(img):
    lines = lsd(img)
    length_segments = np.zeros(lines.shape[0])
    for i in range(lines.shape[0]):
        length_segments[i] = length_segment(lines[i])
    length_segments = np.array(length_segments + 1, dtype=np.int)
    hist, bin_edges = np.histogram(length_segments, bins=int(length_segments.max()) - 1,
                                   range=(1, length_segments.max()))
    return hist, bin_edges


def compute_prob_length(img, thres):
    hist, bin = return_distribution_length(img)
    max_length = int(bin.max())
    hist = hist / hist.sum()
    prob = np.zeros((max_length, max_length))
    min_length_thres = int(thres * max_length) + 1
    for i in range(1, max_length):
        for j in range(min_length_thres, i + 1):
            prob_cond = (bin[j] + (1 - 2 * thres) * bin[i]) / (max_length + 2 * (1 - thres) * bin[i] - bin[j])
            prob[i, j] = prob_cond * hist[i - 1] * hist[j - 1]
    return prob.sum()


def return_NFA(img, tau, thres):
    lines = lsd(img)
    nb_lines = len(lines)
    Ntot = (nb_lines - 1) * (nb_lines - 2) / 2
    return Ntot * tau / 360 * compute_prob_length(img, thres)


if __name__ == '__main__':
    # img_path = "Urben100/img_001_SRF_4_HR.png"
    img_path = "images/square.jpg"
    img = 255 * io.imread(img_path, as_gray=True)
    thres_values = np.array([.1 + .2 * n for n in range(5)])
    probs = np.zeros(5)
    for i in range(5):
        probs[i] = compute_prob_length(img, thres_values[i])
        print(thres_values[i], probs[i])
