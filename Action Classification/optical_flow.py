import os
import sys
import time

import cv2
import numpy as np
from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path


def read_image(path1, path2):
    image1 = cv2.imread(path1)
    if image1 is None:
        sys.exit(-1)
    image2 = cv2.imread(path2)
    if image2 is None:
        sys.exit(-1)
    image1 = cv2.resize(image1, (224, 224))
    image2 = cv2.resize(image2, (224, 224))
    return image1, image2


def posePoints(image1, image2):
    model = "mobilenet_v2_small"
    w = 432
    h = 368
    resize_out_ratio = 3.0

    e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))

    t = time.time()
    humans1 = e.inference(image1, resize_to_default=(w > 0 and h > 0),
                          upsample_size=resize_out_ratio)
    humans2 = e.inference(image2, resize_to_default=(w > 0 and h > 0),
                          upsample_size=resize_out_ratio)
    elapsed = time.time() - t
    print(elapsed)
    return humans1, humans2


def convPoints(humans1, humans2, image1):
    centers1 = []
    centers2 = []
    image_h, image_w = image1.shape[:2]
    for human in humans1:
        # extract point
        for i in range(common.CocoPart.Background.value):
            if i not in human.body_parts.keys():
                continue
            body_part = human.body_parts[i]
            center = [[float(int(body_part.x * image_w + 0.5)),
                       float(int(body_part.y * image_h + 0.5))]]
            centers1.append(list(center))
    for human in humans2:
        # extract point
        for i in range(common.CocoPart.Background.value):
            if i not in human.body_parts.keys():
                continue
            body_part = human.body_parts[i]
            # print(body_part)
            center = [[float(int(body_part.x * image_w + 0.5)),
                       float(int(body_part.y * image_h + 0.5))]]
            centers2.append(list(center))
            # print("centers="+str(centers[i]))
    p0 = np.asarray(centers1, dtype=np.float32)
    p1 = np.asarray(centers2, dtype=np.float32)
    return p0, p1


def generate_optical_mask(image1, image2, p0, p1):
    color = np.random.randint(0, 255, (100, 3))
    frame1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    frame2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(image1)
    pnew, st, err = cv2.calcOpticalFlowPyrLK(frame1, frame2, p0, p1)
    for i, (new, old) in enumerate(zip(p1, p0)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        # frame = cv2.circle(image1,(a,b),5,color[i].tolist(),-1)
    return mask


def dense_optical_flow(img1, img2):
    ## Error on line with #STAR. To resolve the STAR error got the solution on [Link](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&cad=rja&uact=8&ved=2ahUKEwj-jvGy7sTgAhXMTX0KHYMrB34QFjABegQIAxAB&url=https%3A%2F%2Fstackoverflow.com%2Fquestions%2F50319617%2Fopencv-error-cv2-cvtcolor&usg=AOvVaw3bqdEY8piNzb1eJKjjhVTC)
    frame1 = np.array(img1, dtype=np.uint8)  # Solution
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)  # STAR
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    frame2 = np.array(img2, dtype=np.uint8)
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 5, 15, 5, 1, 1.2,
                                        0)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return rgb


def draw_humans(image, humans):
    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
    return image


