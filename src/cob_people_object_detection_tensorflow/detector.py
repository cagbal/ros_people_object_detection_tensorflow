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

from object_detection.utils import ops as utils_ops

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
        output_dict (dictionary) Contains boxes, scores, masks etc.
        """
        with self._detection_graph.as_default():
         # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
               'num_detections', 'detection_boxes', 'detection_scores',
               'detection_classes', 'detection_masks'
               ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                     tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                     tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            start = time.time()

             # Run inference
            output_dict = self._sess.run(tensor_dict,
                     feed_dict={image_tensor: np.expand_dims(image, 0)})

            end = time.time()

            #print end-start

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return (output_dict, self.category_index)

    def visualize(self, image, output_dict):
        """
        Draws the bounding boxes, labels and scores of each detection

        Args:
        image: (numpy array) input image
        output_dict (dictionary) output of object detection model

        Returns:
        image: (numpy array) image with drawings
        """
        # Draw the bounding boxes
        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            output_dict["detection_boxes"],
            output_dict["detection_classes"],
            output_dict["detection_scores"],
            self.category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=5)

        return image
