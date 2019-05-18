import os

import numpy as np
from cv2 import imread, resize

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
        p = os.path.join(path_in, i)
        c = 0
        for j in os.listdir(p):
            m = os.path.join(p, j)
            paths = []
            images = []
            for k in os.listdir(m):
                n = os.path.join(m, k)
                paths.append(n)
            paths = sorted(paths, key=natural_keys)
            image = np.zeros((112, 112, 3))
            for l in paths:
                try:
                    image = resize(imread(l), (112, 112))
                except (KeyboardInterrupt, SystemExit):
                    raise
                except:
                    print(l)
                    pass
                images.append(image)
            video = np.reshape(images, (20, 112, 112, 3))
            print(path_out + i  +'/rgb' + str(c) + '.npy')
            np.save(path_out + i  +'/rgb' + str(c) + '.npy', video)
            c += 1



if __name__ == '__main__':
    path_in = '/home/dgxuser104/Shreyans/data/Delete'
    path_out = '/home/dgxuser104/Shreyans/tf-pose-estimation/har/processed/'
    convert(path_in, path_out)
