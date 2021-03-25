from PIL import Image, ImageDraw
import numpy as np
from pylsd.lsd import lsd
import matplotlib.pyplot as plt
import argparse

from constant_width_detection import constant_width


parser = argparse.ArgumentParser(description='Detection Constant width script')
parser.add_argument('--img_path', type=str, default="images/double_circle.png", metavar='P',
                    help='path of the image')
parser.add_argument('--tau', type=float, default=10, metavar='TA',
                    help='parameter for parallelism')
parser.add_argument('--thres', type=float, default=.5, metavar='TH',
                    help='parameter for overlapping')
parser.add_argument('--NFA', type=float, default=1, metavar='N',
                    help='Number of false alerts')
parser.add_argument('--compute_param', type=bool, default=False, metavar='C',
                    help='If True will compute the thres such that NFA is around 1')


args = parser.parse_args()

img = Image.open(args.img_path)

mode = 'RGB'  # for color image “L” (luminance) for greyscale images, “RGB” for true color images
size = img.size[:2]
color = (255, 255, 255)

im = Image.new(mode, size, color)
im_2 = Image.new(mode, size, color)

gray = np.asarray(img.convert('L'))
lines = lsd(gray)
# shape nb_lines 5 ([point1.x, point1.y, point2.x, point2.y, width])
draw = ImageDraw.Draw(im)
# draw lines
draw_2 = ImageDraw.Draw(im_2)
# draw constant width

index_matching_lines = constant_width(gray, tau=args.tau, thres=args.thres, compute_param=args.compute_param)
nb_pairs = len(index_matching_lines)

# show detected constant width
for pair_matching in range(nb_pairs):
    i, j = index_matching_lines[pair_matching]
    pt1 = (int(lines[i, 0]), int(lines[i, 1]))
    pt2 = (int(lines[i, 2]), int(lines[i, 3]))
    pt3 = (int(lines[j, 0]), int(lines[j, 1]))
    pt4 = (int(lines[j, 2]), int(lines[j, 3]))
    pts = (pt1, pt2, pt3, pt4)

    c1, c2, c3 = np.random.randint(0, 200), np.random.randint(0, 255), np.random.randint(0, 255)
    draw_2.polygon(pts, fill=(c1, c2, c3))

    draw_2.line((pt1, pt2), fill=(255, 0, 0), width=1)
    draw_2.line((pt3, pt4), fill=(255, 0, 0), width=1)

# show detected lines
for i in range(lines.shape[0]):
    pt1 = (int(lines[i, 0]), int(lines[i, 1]))
    pt2 = (int(lines[i, 2]), int(lines[i, 3]))
    width = lines[i, 4]
    draw.line((pt1, pt2), fill=(255, 0, 0), width=1)


if __name__ == '__main__':
    plt.axis("off")
    plt.imshow(img)
    plt.show()

    plt.axis("off")
    plt.imshow(im)
    plt.show()

    plt.axis("off")
    plt.imshow(im_2)
    plt.show()
