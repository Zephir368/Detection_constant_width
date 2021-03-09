from PIL import Image, ImageDraw
import numpy as np
from pylsd.lsd import lsd
import matplotlib.pyplot as plt
import skimage.io as io
import seaborn as sns

from aligned_segments import length_segment

from constant_width_detection import constant_width

img_path = "Urben100/img_002_SRF_4_HR.png"
# img_path = "images/chairs.png"

img = 255 * io.imread(img_path, as_gray=True)

lines = lsd(img)
length_segments = np.zeros(lines.shape[0])

for i in range(lines.shape[0]):
    length_segments[i] = length_segment(lines[i])

sns.set_theme()
sns.histplot(data=length_segments/680, kde=True, stat="density")
plt.show()
