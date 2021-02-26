from PIL import Image, ImageDraw
import numpy as np
from pylsd.lsd import lsd
import matplotlib.pyplot as plt

from constant_width_detection import constant_width, return_middle, compute_parallel_segments



#img_path = "Urben100/img_001_SRF_4_HR.png"
img_path = "images/square.jpg"
img = Image.open(img_path)


mode = 'RGB' # for color image “L” (luminance) for greyscale images, “RGB” for true color images, and “CMYK” for pre-press images.
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


detected_constant = constant_width(gray, lines, 20)
nb_pairs = len(detected_constant)


for pair_matching in range(nb_pairs):
    print(pair_matching)
    i, j = detected_constant[pair_matching]
    pt1 = (int(lines[i, 0]), int(lines[i, 1]))
    pt2 = (int(lines[i, 2]), int(lines[i, 3]))
    pt3 = (int(lines[j, 2]), int(lines[j, 3]))
    pt4 = (int(lines[j, 0]), int(lines[j, 1]))
    
    width = lines[i, 4]
    draw_2.line((pt1, pt2), fill=(0, 0, 255), width=int(np.ceil(width / 2)))
    width = lines[j, 4]
    draw_2.line((pt3, pt4), fill=(0, 0, 255), width=int(np.ceil(width / 2)))

    draw_2.line((pt1, pt3), fill=(255, 0, 0), width=int(np.ceil(width / 2)))
    draw_2.line((pt2, pt4), fill=(255, 0, 0), width=int(np.ceil(width / 2)))


for i in range(lines.shape[0]):
    pt1 = (int(lines[i, 0]), int(lines[i, 1]))
    pt2 = (int(lines[i, 2]), int(lines[i, 3]))
    width = lines[i, 4]
    draw.line((pt1, pt2), fill=(int(i * 255 / 7), 0, 0), width=int(np.ceil(width / 2)))


draw.line(((0,0), (200,200)), fill=(255, 0, 0), width=int(np.ceil(width / 2)))

plt.axis("off")
plt.imshow(img)
plt.show()

plt.axis("off")
plt.imshow(im)
plt.show()

plt.axis("off")
plt.imshow(im_2)
plt.show()
