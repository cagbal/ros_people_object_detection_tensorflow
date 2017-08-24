#!/usr/bin/env python

"""
Detector class to use TensorFlow detection API

The codes are from TensorFlow/Models Repo. I just transferred the code to
ROS.

Cagatay Odabasi
"""

import numpy as np
import tensorflow as tf
import time

import cv2

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import rospkg

class Detector(object):
    """docstring for Detector."""
    def __init__(self, \
        model_name='ssd_mobilenet_v1_coco_11_06_2017',\
        num_of_classes=90,\
        label_file='mscoco_label_map.pbtxt'\
        ):

        super(Detector, self).__init__()
        # What model to download.
        self._model_name = model_name
        # ssd_inception_v2_coco_11_06_2017

        self._num_classes = num_of_classes

        self._detection_graph = None

        self._sess = None

        self.category_index = None

        self._label_file = label_file

        # get an instance of RosPack with the default search paths
        rospack = rospkg.RosPack()

        self._tf_object_detection_path = \
            rospack.get_path('cob_people_object_detection_tensorflow') + \
            '/src/object_detection'

        self._path_to_ckpt = self._tf_object_detection_path + '/' + \
            self._model_name + '/frozen_inference_graph.pb'

        # List of the strings that is used to add correct label for each box.
        self._path_to_labels = self._tf_object_detection_path + '/' + \
            'data/' + self._label_file

        # Prepare the model for detection
        self.prepare()

    def load_model(self):
        """
        Loads the detection model

        Args:

        Returns:

        """

        with self._detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self._path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        label_map = label_map_util.load_labelmap(self._path_to_labels)
        categories = label_map_util.convert_label_map_to_categories(\
            label_map, max_num_classes=self._num_classes, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

    def prepare(self):
        """
        Prepares the model for detection

        Args:

        Returns:

        """

        self._detection_graph = tf.Graph()

        self.load_model()

        self._sess = tf.Session(graph=self._detection_graph)


    def detect(self, image):
        """
        Detects objects in the image given

        Args:
        image: (numpy array) input image

        Returns:
        (boxes, score, classes)
        boxes: Bounding boxes for each object detection
        score: Confidence score of each object detection
        classes: (integer) class labels of each detection

        """

        with self._detection_graph.as_default():

            # Expand dimensions since the model expects
            # images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image, axis=0)
            image_tensor = \
               self._detection_graph.get_tensor_by_name('image_tensor:0')
            # Bounding boxes for each object detection
            boxes = \
                self._detection_graph.get_tensor_by_name('detection_boxes:0')
            # Confidence scores
            scores = \
                self._detection_graph.get_tensor_by_name('detection_scores:0')
            classes = \
               self._detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = \
               self._detection_graph.get_tensor_by_name('num_detections:0')

            start = time.time()

            # Detect
            (boxes, scores, classes, num_detections) = self._sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            end = time.time()

            print end-start

            return (boxes, scores, classes, self.category_index)

    def visualize(self, image, boxes, scores, classes):
        """
        Draws the bounding boxes, labels and scores of each detection

        Args:
        image: (numpy array) input image
        boxes: Bounding boxes for each object detection
        score: Confidence score of each object detection
        classes: (integer) class labels of each detection

        Returns:
        image: (numpy array) image with drawings 
        """
        # Draw the bounding boxes
        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=5)

        return image
