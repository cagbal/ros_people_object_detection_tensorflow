#!/usr/bin/env python
"""
A ROS node to detect objects via TensorFlow Object Detection API.

Author:
    Cagatay Odabasi -- cagatay.odabasi@ipa.fraunhofer.de
"""

# ROS
import rospy

from cob_perception_msgs.msg import Detection, DetectionArray, Rect

from sensor_msgs.msg import Image

from cv_bridge import CvBridge, CvBridgeError

from cob_people_object_detection_tensorflow.detector import Detector

from cob_people_object_detection_tensorflow import utils

class PeopleObjectDetectionNode(object):
    """docstring for PeopleObjectDetectionNode."""
    def __init__(self):
        super(PeopleObjectDetectionNode, self).__init__()

        # init the node
        rospy.init_node('people_object_detection', anonymous=False)

        # Get the parameters
        (model_name, num_of_classes, label_file, camera_namespace) = \
            self.get_parameters()

        # Create Detector
        self._detector = Detector(model_name, num_of_classes, label_file)

        self._bridge = CvBridge()

        # Advertise the result of Object Detector
        self.pub_detections = rospy.Publisher('/object_detection/detections', \
            DetectionArray, queue_size=1)

        # Advertise the result of Object Detector
        self.pub_detections_image = rospy.Publisher(\
            '/object_detection/detections_image', Image, queue_size=1)

        # Subscribe to the face positions
        self.sub_rgb = rospy.Subscriber(camera_namespace,\
            Image, self.rgb_callback, queue_size=1, buff_size=2**24)

        # spin
        rospy.spin()

    def get_parameters(self):
        """
        Gets the necessary parameters from parameter server

        Args:

        Returns:
        (tuple) (model name, num_of_classes, label_file)

        """

        model_name  = rospy.get_param("~model_name")
        num_of_classes  = rospy.get_param("~num_of_classes")
        label_file  = rospy.get_param("~label_file")
        camera_namespace  = rospy.get_param("~camera_namespace")

        return (model_name, num_of_classes, label_file, camera_namespace)


    def shutdown(self):
        """
        Shuts down the node
        """
        rospy.signal_shutdown("See ya!")

    def rgb_callback(self, data):
        """
        Callback for RGB images
        """
        try:
            # Conver image to numpy array
            cv_image = self._bridge.imgmsg_to_cv2(data, "bgr8")

            # Detect
            (boxes, scores, classes, category_index) = \
                self._detector.detect(cv_image)

            # Create the message
            msg = \
                utils.create_detection_msg(\
                data, boxes, scores, classes, category_index)

            # Draw bounding boxes
            image_processed = \
                self._detector.visualize(cv_image, boxes, scores, classes)

            # Convert numpy image into sensor img
            msg_im = \
                self._bridge.cv2_to_imgmsg(\
                image_processed, encoding="passthrough")

            # Publish the messages
            self.pub_detections.publish(msg)
            self.pub_detections_image.publish(msg_im)

        except CvBridgeError as e:
            print(e)

def main():
    """ main function
    """
    node = PeopleObjectDetectionNode()

if __name__ == '__main__':
    main()
