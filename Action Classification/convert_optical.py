import os

import numpy as np
from cv2 import imread, resize
from optical_flow import * 
from common import classes
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def convert(path_in, path_out):
    for i in classes:
        path = os.path.join(path_in, i)
        path_optical = os.path.join(path_out, i)
        if not os.path.exists(path_optical):
            os.makedir(path_optical)
        for j in range(100):
            rgb_file = 'rgb' + j + '.npy'
            optical_file = 'flow' + j + '.npy'
            print(os.path.join(path, rgb_file))



if __name__ == '__main__':
    path_in = 'processed'
    path_out = 'processed_flow'
    convert(path_in, path_out)
