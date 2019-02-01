import scipy.misc
import matplotlib.image as mpimg
import numpy as np
import sys
import os

def upsample4x(image_arr, interp):
    resized_image = scipy.misc.imresize(image_arr, 4.0, interp=interp, mode=None)
    return resized_image

if (len(sys.argv) != 4):
    print("need input and output directory")
    exit()

input_dir = str(sys.argv[1])
output_dir = str(sys.argv[2])
interp = str(sys.argv[3])

for filename in os.listdir(input_dir):
    if filename.endswith(".png"):
        img = mpimg.imread(input_dir + '/' + filename)
        img = np.asarray(img, dtype=np.float64)
        new_img = upsample4x(img, interp)
        mpimg.imsave(output_dir + '/' + filename, new_img)
