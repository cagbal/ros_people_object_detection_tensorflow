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

import cv2

from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image, PointCloud2

from cob_perception_msgs.msg import Detection, DetectionArray, Rect

from pcl_helper import *

from skimage import filters




class ProjectionNode(object):
    """docstring for ProjectionNode."""
    def __init__(self):
        super(ProjectionNode, self).__init__()

        # init the node
        rospy.init_node('projection_node', anonymous=False)

        self._bridge = CvBridge()

        (depth_topic, face_topic, output_topic, f) = self.get_parameters()

         # Subscribe to the face positions
        sub_obj = message_filters.Subscriber(face_topic ,\
            DetectionArray)

        sub_depth = message_filters.Subscriber(depth_topic,\
            Image)

        # Advertise the result of Face Depths
        self.pub = rospy.Publisher(output_topic, \
            DetectionArray, queue_size=1)

        ts = message_filters.ApproximateTimeSynchronizer([sub_obj, sub_depth], 2, 0.9)

        ts.registerCallback(self.detection_callback)

        (depth_topic, face_topic, output_topic, f) = self.get_parameters()

        self.f = f

        # spin
        rospy.spin()


    def shutdown(self):
        """
        Shuts down the node
        """
        rospy.signal_shutdown("See ya!")

    def detection_callback(self, msg, depth):
        """
        Callback for RGB images
        Args:
        detections: detections array message from cob package
        point_cloud (sensor_msgs/PointCloud2) point cloud input from camera

        """

        cv_depth = self._bridge.imgmsg_to_cv2(depth, "passthrough")

        # get the number of detections
        no_of_detections = len(msg.detections)

        print("hey")

        # Check if there is a detection
        if no_of_detections > 0:
            for i, detection in enumerate(msg.detections):

                label = detection.label
                confidence = detection.score

                x =  detection.mask.roi.x
                y = detection.mask.roi.y
                width =  detection.mask.roi.width
                height = detection.mask.roi.height

                cv_depth = cv_depth[y:y+height,x:x+width]

                depth_mean = np.nanmedian(cv_depth[np.nonzero(cv_depth)])

                try:

                    depth_mean = np.nanmedian(cv_depth[np.nonzero(cv_depth)])


                    real_x = (x + width/2-320)*self.f*depth_mean

                    real_y = (y + height/2-240)*self.f*depth_mean

                    msg.detections[i].pose.pose.position.x = real_x
                    msg.detections[i].pose.pose.position.y = real_y
                    msg.detections[i].pose.pose.position.z = depth_mean

                except Exception as e:
                    print e

        self.pub.publish(msg)

    def get_parameters(self):
        """
        Gets the necessary parameters from parameter server

        """

        depth_topic  = rospy.get_param("~depth_topic")
        face_topic = rospy.get_param('~face_topic')
        output_topic = rospy.get_param('~output_topic')
        f = rospy.get_param('~focal_length')

        return (depth_topic, face_topic, output_topic, f)



def main():
    """ main function
    """
    node = ProjectionNode()

if __name__ == '__main__':
    main()
