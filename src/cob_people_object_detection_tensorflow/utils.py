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

def create_detection_msg(im, output_dict, category_index, bridge):
    """
    Creates the detection array message

    Args:
    im: (std_msgs_Image) incomming message

    output_dict (dictionary) output of object detection model

    category_index: dictionary of labels (like a lookup table)

    bridge (cv_bridge) : cv bridge object for converting

    Returns:

    msg (cob_perception_msgs/DetectionArray) The message to be sent

    """

    boxes = output_dict["detection_boxes"]
    scores = output_dict["detection_scores"]
    classes = output_dict["detection_classes"]
    masks = None

    if 'detection_masks' in output_dict:
        masks = output_dict["detection_masks"]

    msg = DetectionArray()

    msg.header = im.header

    scores_above_threshold = np.where(scores > 0.5)[0]

    for s in scores_above_threshold:
        # Get the properties

        bb = boxes[s,:]
        sc = scores[s]
        cl = classes[s]

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

        if 'detection_masks' in output_dict:
            detection.mask.mask = \
                bridge.cv2_to_imgmsg(masks[s], "mono8")

            print detection.mask.mask.width


        msg.detections.append(detection)

    return msg
