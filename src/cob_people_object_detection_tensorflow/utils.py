#!/usr/bin/env python

"""
Helper functions and classes will be placed here.
"""

import os
import tarfile
import six.moves.urllib as urllib

import numpy as np

from cob_perception_msgs.msg import Detection, DetectionArray, Rect


def download_model(\
    download_base='http://download.tensorflow.org/models/object_detection/', \
    model_name='ssd_mobilenet_v1_coco_11_06_2017'\
    ):
    """
    Downloads the detection model from tensorflow servers

    Args:
    download_base: base url where the object detection model is downloaded from

    model_name: name of the object detection model

    Returns:

    """

    # add tar gz to the end of file name
    model_file = model_name + '.tar.gz'

    try:
        opener = urllib.request.URLopener()
        opener.retrieve(download_base + model_file, \
            model_file)
        tar_file = tarfile.open(model_file)
        for f in tar_file.getmembers():
            file_name = os.path.basename(f.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(f, os.getcwd())
    except Exception as e:
        raise

def create_detection_msg(im, boxes, scores, classes, category_index):
    """
    Creates the detection array message

    Args:
    im: (std_msgs_Image) incomming message

    boxes: bounding boxes of objects

    scores: scores of each bb

    classes: class predictions

    category_index: dictionary of labels (like a lookup table)

    Returns:

    msg (cob_perception_msgs/DetectionArray) The message to be sent

    """

    msg = DetectionArray()

    msg.header = im.header

    scores_above_threshold = np.where(scores > 0.5)[1]

    for s in scores_above_threshold:
        # Get the properties
        bb = boxes[0,s,:]
        sc = scores[0,s]
        cl = classes[0,s]

        print category_index

        # Create the detection message
        detection = Detection()
        detection.header = im.header
        detection.label = category_index[int(cl)]['name']
        detection.id = cl
        detection.score = sc
        detection.detector = 'Tensorflow object detector'
        detection.mask.roi.x = int((im.width-1) * bb[1])
        detection.mask.roi.y = int((im.height-1) * bb[0])
        detection.mask.roi.width = int((im.width-1) * (bb[3]-bb[1]))
        detection.mask.roi.height = int((im.height-1) * (bb[2]-bb[0]))

        msg.detections.append(detection)

    return msg






    return msg
