import os
import numpy as np
import cv2
from utilities.utils import *


def image_out(args, input_img, indexes):
    '''
    takes similar image index as input and returns 
    a grid of images for better visualization
    '''
    lst = []
    names = open(os.path.join(args.root_dir, 'img_names.txt'), 'r').read().splitlines()
    for i in indexes:
        img = cv2.imread(os.path.join(args.img_folder, names[i-1]), -1)
        img = remove_dummy(img)  
        lst.append(img)

    horz1 = np.concatenate((lst[0], lst[1], lst[2], lst[3], lst[4]), axis=1)
    horz2 = np.concatenate((lst[5], lst[6], lst[7], lst[8], lst[9]), axis=1)
    vert = np.concatenate((horz1, horz2), axis=0)

    out = np.concatenate((input_img, vert), axis=1)

    return out