#!/usr/bin/env python
"""
A ROS node to detect objects via TensorFlow Object Detection API.

Author:
    Cagatay Odabasi -- cagatay.odabasi@ipa.fraunhofer.de
"""

import time

import rospy

import message_filters

import numpy

from scipy.optimize import linear_sum_assignment

import cv2

from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image

from cob_perception_msgs.msg import Detection, DetectionArray, Rect

from cob_people_object_detection_tensorflow.detector import Detector

from cob_people_object_detection_tensorflow import utils

from sort import sort


class PeopleObjectTrackerNode(object):
    """docstring for PeopleObjectDetectionNode."""
    def __init__(self):
        super(PeopleObjectTrackerNode, self).__init__()

        # init the node
        rospy.init_node('people_object_tracker', anonymous=False)

        # Get the parameters
        (camera_topic, detection_topic, tracker_topic, cost_threshold) = \
            self.get_parameters()

        self._bridge = CvBridge()

        self.tracker =  sort.Sort()

        self.cost_threshold = cost_threshold

        # Subscribe to the face positions
        sub_detection = message_filters.Subscriber(detection_topic, \
            DetectionArray)

        sub_rgb = message_filters.Subscriber(camera_topic, Image)

        # Advertise the result of Object Tracker
        self.pub_detections = rospy.Publisher(tracker_topic, \
            DetectionArray, queue_size=1)

        ts = message_filters.ApproximateTimeSynchronizer([sub_detection, \
            sub_rgb], 2, 0.20)

        ts.registerCallback(self.detection_callback)

        # spin
        rospy.spin()

    def get_parameters(self):
        """
        Gets the necessary parameters from parameter server

        Args:

        Returns:
        (tuple) (camera_topic, detection_topic, tracker_topic, cost_threhold)

        """

        camera_topic  = rospy.get_param("~camera_topic")
        detection_topic  = rospy.get_param("~detection_topic")
        tracker_topic = rospy.get_param('~tracker_topic')
        cost_threhold = rospy.get_param('~cost_threhold')

        return (camera_topic, detection_topic, tracker_topic, cost_threhold)


    def shutdown(self):
        """
        Shuts down the node
        """
        rospy.signal_shutdown("See ya!")

    def detection_callback(self, detections, image):
        """
        Callback for RGB images
        Args:
        detections: detections array message from cob package

        image: rgb frame in openCV format

        """

        cv_rgb = self._bridge.imgmsg_to_cv2(image, "passthrough")

        # Dummy detection
        # TODO: Find a better way
        det_list = numpy.array([[0, 0, 1, 1, 0.01]])

        if len(detections.detections) > 0:
            for detection in detections.detections:
                x =  detection.mask.roi.x
                y = detection.mask.roi.y
                width =  detection.mask.roi.width
                height = detection.mask.roi.height
                score = detection.score

                det_list = numpy.vstack((det_list, \
                    [x, y, width, height, score]))

        # Call the tracker
        tracks = self.tracker.update(det_list, cv_rgb)


        if len(det_list) > 1:

            # Create cost matrix
            # Double for in Python :(

            C = numpy.zeros((len(tracks), len(det_list)))
            for i, track in enumerate(tracks):
                for j, det in enumerate(det_list):
                    C[i, j] = numpy.linalg.norm(det[0:-2] - track[0:-2])


            # apply linear assignment
            row_ind, col_ind = linear_sum_assignment(C)

            for i, j in zip(row_ind, col_ind):
                if C[i, j] < self.cost_threshold and j != 0:
                    print("{} -> {} with cost {}".\
                        format(tracks[i, 4], detections.detections[j-1].label,\
                        C[i,j]))
        print "hey"


def main():
    """ main function
    """
    node = PeopleObjectTrackerNode()

if __name__ == '__main__':
    main()
