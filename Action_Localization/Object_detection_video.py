######## Video Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/16/18
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on a video.
# It draws boxes and scores around the objects of interest in each frame
# of the video.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

# Import packages
import os
import cv2
from cv2 import imread, resize
import numpy as np
import tensorflow as tf
import sys
import time
from action_classification_models import C3D
from Classes_array_common import classes
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'
#VIDEO_NAME = 'S010C001P007R001A060_rgb.avi' #walk appart
#VIDEO_NAME = 'S013C001P007R001A057_rgb.avi' #theft
#VIDEO_NAME = 'S013C003P019R001A052_rgb.avi' #Pushing ## Worng output
VIDEO_NAME = 'S013C001P017R002A018_rgb.avi' #Wearing glasses
#VIDEO_NAME = 'S013C001P018R001A041_rgb.avi' #Sneazing/coughing

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Path to video
PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 1

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

def frameCount(video):
    #________________________________________________________________________________
    #Count the number of frames
    #________________________________________________________________________________
    no_of_frames=0
    #Read the video
    ret_val, image = video.read()
    while ret_val:
        ret_val, image = video.read()
        no_of_frames+=1
    return no_of_frames

def convertVideotoNPY(video_path,no_of_frames):
    #anubhav
    images=[]
    #Read the file
    video = cv2.VideoCapture(video_path)
    ret_val, image = video.read()

    #________________________________________________________________________________
    #If number of frames is less than 40
    #________________________________________________________________________________
    #36+x=40, x=40-36 = 4 (overwrite the frame)
    if(no_of_frames<40):
        count=0
        sampling_rate=2 #divide 40 by 2 so we get 20 frames
        frame_cut=40-no_of_frames #number of frames to be added
        i=0

        while ret_val:
            if(i<frame_cut):
                ret_val, image = video.read()
                frameno=count
                count+=1
                i=i+1
                continue
            else:
                i=i+1
                ret_val, image = video.read()
                frameno=count
                img = np.zeros((112, 112, 3))
                if(count%sampling_rate==0):
                    img = resize(image, (112, 112))
                    images.append(img)
                    #cv2.imshow('tf-pose-estimation result', image)
                    #cv2.imwrite(output_path+"\\frame%d.jpg" % frameno, image)
                count+=1

    #__________________________________________________________________________________
    #Trimming Number of Frames <80
    #__________________________________________________________________________________
    #If number of frame is less than 80 means its 1 second video, so cut the number of frames from the
    #beginning as those are mostly stable frames (action is happening in later frames).

    #Read the video again as after reading it once and calling it, the variable becomes empty
    #cam = cv2.VideoCapture(video_path)
    #ret_val, image = cam.read()
    #58-x=40, x=58-40 = 12 (Maths, so total number of frames become equal to 40 for 1 second video)
    elif(no_of_frames<80):
        count=0
        sampling_rate=2 #divide 40 by 2 so we get 20 frames
        frame_cut=no_of_frames-40 #number of frames to be trimmed
        i=0

        #Counting the number of frames which we are skipping
        while ret_val:
            if(i<frame_cut):
                ret_val, image = video.read()
                frameno=count
                count+=1
                i=i+1
                continue
            else:
                i=i+1
                ret_val, image = video.read()
                frameno=count
                img = np.zeros((112, 112, 3))
                if(count%sampling_rate==0):
                    img = resize(image, (112, 112))
                    images.append(img)
                    #cv2.imshow('tf-pose-estimation result', image)
                    #cv2.imwrite(output_path+"\\frame%d.jpg" % frameno, image)
                count+=1

    #__________________________________________________________________________________
    #Trimming Number of Frames <120
    #__________________________________________________________________________________
    #58-x=80, x=58-80 (Maths, so total number of frames become equal to 80 for 2 second video)
    elif(no_of_frames<120):
        count=0
        sampling_rate=4
        frame_cut=no_of_frames-80 #divide 80 by 4 so we get 20 frames
        i=0
        while ret_val:
            if(i<frame_cut):
                ret_val, image = video.read()
                frameno=count
                count+=1
                i=i+1
                continue
            else:
                i=i+1
                ret_val, image = video.read()
                frameno=count
                img = np.zeros((112, 112, 3))
                if(count%sampling_rate==0):
                    img = resize(image, (112, 112))
                    images.append(img)
                    #cv2.imshow('tf-pose-estimation result', image)
                    #cv2.imwrite(output_path+"\\frame%d.jpg" % frameno, image)
                count+=1

    #__________________________________________________________________________________
    #Trimming Number of Frames >=120
    #__________________________________________________________________________________
    #58-x=120, x=58-120 (Maths, so total number of frames become equal to 120 for 3 second video)
    elif(no_of_frames>=120):
        count=0
        sampling_rate=6 #divide 120 by 6 so we get 30 frames 
        frame_cut=no_of_frames-120
        i=0
        while ret_val:
            if(i<frame_cut):
                ret_val, image = video.read()
                frameno=count
                count+=1
                i=i+1
                continue
            else:
                i=i+1
                ret_val, image = video.read()
                frameno=count
                img = np.zeros((112, 112, 3))
                if(count%sampling_rate==0):
                    img = resize(image, (112, 112))
                    images.append(img)
                    #cv2.imshow('tf-pose-estimation result', image)
                    #cv2.imwrite(output_path+"\\frame%d.jpg" % frameno, image)
                count+=1

    #___________________________________________________________________
    #Shaurya ka code for converting all the frames to npy format
    
    video = np.reshape(images, (20, 112, 112, 3))
    np.save('rgb' + '.npy', video)

#------------------------------------------------------------------------------------

def noOfIntersect(classes):
    cou=0
    for i in range(0,int(num)):
        if (classes[0][i]==1):
            cou+=1
    return cou

def boundingBox(boxes,classes):
    noi=noOfIntersect(classes)
    x,y=[],[]
    for i in range(0,noi):
        for j in range(0,4):
            if(j%2==0):
                y.append(boxes[i,j])
            else:
                x.append(boxes[i,j])
    #print(x,y)
    return x,y

def makeBox(x,y):
    xmin=min(list(x))
    ymin=min(list(y))
    xmax=max(list(x))
    ymax=max(list(y))
    #print(ymin,xmin,ymax,xmax)
    return ymin,xmin,ymax,xmax
    

def convertToArray(boxes,classes,num):
    noi=noOfIntersect(classes)
    a=[[0]*4]*noi
    ind=0  
    for i in range(0,int(num)):
        if(classes[0][i]==1):
            ##print(boxes[0][i][:],"   ",ind)
            a[ind]=list((boxes[0][i][:]))
            ind+=1
    ##print("A :",a)      
    b=np.asarray(a)
    ##print("Arr \n",b)
    return b

def calcScore(scores,classes):
    noi= noOfIntersect(classes)
    ##print(noi)
    score_max = []
    for i in range (0, int(num)):
        if classes[0][i]==1:
            score_max.append(scores[0][i])
    return max(score_max)





# Open video file
video = cv2.VideoCapture(PATH_TO_VIDEO)
noOfFrames = frameCount(video)

video = cv2.VideoCapture(PATH_TO_VIDEO)
count=0
while(video.isOpened() and count<noOfFrames-1):
    ret, frame = video.read()
    time.sleep(1/20)
    # Display the resulting frame
    cv2.imshow('frame',frame)
    count=count+1
    if cv2.waitKey(3) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

convertVideotoNPY(PATH_TO_VIDEO,noOfFrames)

#Classification Prediction
classification_model = C3D()
classification_model.load_weights('C:\\tensorflow1\\models\\research\\object_detection\\action_classification_saved\\model175.hdf5')
v = np.load('rgb.npy')[:, :, :, ::-1]
p = [v for i in range(1)]
p = np.reshape(p, newshape=(1, 20, 112, 112, 3))
y = classification_model.predict(p)
y = np.sum(y, axis=0)
class_label = str(classes[np.argmax(y / 1)])
print(class_label)

#class_label = "action"

# Detection Prediction
count=0
video = cv2.VideoCapture(PATH_TO_VIDEO)

while(video.isOpened() and count<noOfFrames-1):

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = video.read()
    #cv2.imshow('Video', frame)
    frame_expanded = np.expand_dims(frame, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})
    
    arr=convertToArray(boxes,classes,num)
    x,y=boundingBox(arr,classes)
    ymin,xmin,ymax,xmax=makeBox(x,y)
    carr=np.array([[ymin,xmin,ymax,xmax]])
    claes=np.array([1])
    score=np.array([calcScore(scores,classes)])
    #print(carr,claes,score)

    #Draw the results of the detection (aka 'visulaize the results')
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        carr,
        claes.astype(np.int32),
        score,
        category_index,
        new_label=class_label,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0)
    
    # vis_util.visualize_boxes_and_labels_on_image_array(
    #     frame,
    #     np.squeeze(boxes),
    #     np.squeeze(classes).astype(np.int32),
    #     np.squeeze(scores),
    #     category_index,
    #     new_label=class_label,
    #     use_normalized_coordinates=True,
    #     line_thickness=8,
    #     min_score_thresh=0)

    #Saving frames after anotating them
    # cv2.imwrite("output//UnFinishedframe%d.jpg" % count, frame)
    count+=1

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break
        


# Clean up
video.release()
cv2.destroyAllWindows()
