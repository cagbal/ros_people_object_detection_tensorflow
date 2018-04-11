#!/usr/bin/env python
"""
A ROS node to get face bounding boxes inside of person bounding boxes returned
by object detection node.

This node gets the people bounding boxes and applies face searching and face
comparison by using face_recognition Python library.

The people who are placed in /people directory will be automatically fetched
and their face features will be compared to incoming face images. If a similar
face is found, the name of the closest face image will be assigned to that
bounding box.

Author:
    Cagatay Odabasi -- cagatay.odabasi@ipa.fraunhofer.de
"""

import glob

import rospy

import message_filters

import numpy as np

import cv2

from cv_bridge import CvBridge

from sensor_msgs.msg import Image

from cob_perception_msgs.msg import DetectionArray

import face_recognition as fr

import rospkg


# Get the package directory
rospack = rospkg.RosPack()

cd = rospack.get_path('cob_people_object_detection_tensorflow')


class FaceRecognitionNode(object):
    """A ROS node to get face bounding boxes inside of person bounding boxes

    _bridge (CvBridge): Bridge between ROS and CV image
    pub_det (Publisher): Publisher object for detections (bounding box, labels)
    pub_det_rgb (Publisher): Publisher object for detection image
    sub_detection (Subscriber): Subscriber object for object_detection
    sub_image (Subscriber): Subscriber object for RGB image from camera
    scaling_factor (Float): Input image will be scaled down with this
    database (List): Contains face features of people inside /people folder

    """
    def __init__(self):
        super(FaceRecognitionNode, self).__init__()

        # init the node
        rospy.init_node('face_recognition_node', anonymous=False)

        # Get the parameters
        (image_topic, detection_topic, output_topic, output_topic_rgb) \
            = self.get_parameters()

        self._bridge = CvBridge()

        # Advertise the result of Object Tracker
        self.pub_det = rospy.Publisher(output_topic, \
            DetectionArray, queue_size=1)

        self.pub_det_rgb = rospy.Publisher(output_topic_rgb, \
            Image, queue_size=1)


        self.sub_detection = message_filters.Subscriber(detection_topic, \
            DetectionArray)
        self.sub_image = message_filters.Subscriber(image_topic, Image)

        # Scaling factor for face recognition image
        self.scaling_factor = 0.50

        # Read the images from folder and create a database
        self.database = self.initialize_database()

        ts = message_filters.ApproximateTimeSynchronizer(\
            [self.sub_detection, self.sub_image], 2, 0.2)

        ts.registerCallback(self.detection_callback)

        # spin
        rospy.spin()

    def get_parameters(self):
        """
        Gets the necessary parameters from parameter server

        Args:

        Returns:
        (tuple) (camera_topic, detection_topic, output_topic)

        """

        camera_topic = rospy.get_param("~camera_topic")
        detection_topic = rospy.get_param("~detection_topic")
        output_topic = rospy.get_param("~output_topic")
        output_topic_rgb = rospy.get_param("~output_topic_rgb")

        return (camera_topic, detection_topic, output_topic, output_topic_rgb)


    def shutdown(self):
        """
        Shuts down the node
        """
        rospy.signal_shutdown("See ya!")

    def detection_callback(self, detections, image):
        """
        Callback for RGB images and detections

        Args:
        detections (cob_perception_msgs/DetectionArray) : detections array
        image (sensor_msgs/Image): RGB image from camera

        """

        cv_rgb = self._bridge.imgmsg_to_cv2(image, "passthrough")[:, :, ::-1]

        cv_rgb = cv2.resize(cv_rgb, (0, 0), fx=self.scaling_factor, fy=self.scaling_factor)

        cv_rgb=cv_rgb.astype(np.uint8)

        (cv_rgb, detections) = self.recognize(detections, cv_rgb)

        image_outgoing = self._bridge.cv2_to_imgmsg(cv_rgb, encoding="passthrough")

        self.publish(detections, image_outgoing)


    def recognize(self, detections, image):
        """
        Main face recognition logic, it gets the incoming detection message and
        modifies the person labeled detections according to the face info.

        For example:
         (label) person, bounding_box -> (label) Mario, bounding_box of face

        Args:
        (DetectionArray) detections: detections array message from cob package
        (numpy.ndarray) image: incoming people image

        Returns:

        (numpy.ndarray): image with labels and bounding boxes
        (cob_perception_msgs/DetectionArray): detections with labeled faces

        """

        detections_out = DetectionArray()
        detections_out.header = detections.header

        for i, detection in enumerate(detections.detections):

            if detection.label == "person":

                x =  int(detection.mask.roi.x * self.scaling_factor)
                y = int(detection.mask.roi.y * self.scaling_factor)
                width = int(detection.mask.roi.width * self.scaling_factor)
                height = int(detection.mask.roi.height * self.scaling_factor)
                score = detection.score

                try:
                    # Crop detection image
                    detection_image = image[x:x+width, y:y+height]

                    face_locations = fr.face_locations(detection_image)

                    face_features = fr.face_encodings(detection_image, \
                        face_locations)

                    for features, (top, right, bottom, left) in \
                        zip(face_features, face_locations):
                        matches = fr.compare_faces(self.database[0], features)

                        l = y + left
                        t = x + top
                        r = y + right
                        b = x + bottom

                        detection.label = "Unknown"

                        if True in matches:
                            ind = matches.index(True)
                            detection.label = self.database[1][ind]

                        # Modify the message
                        detection.mask.roi.x  = l/self.scaling_factor
                        detection.mask.roi.y = t/self.scaling_factor
                        detection.mask.roi.width = (r-l)/self.scaling_factor
                        detection.mask.roi.height = (b-t)/self.scaling_factor

                        # Draw bounding boxes on current image

                        cv2.rectangle(image, (l, t), \
                        (r, b), (0, 0, 255), 2)

                        cv2.rectangle(image, (x, y), \
                        (x + width, y + height), (255, 0, 0), 3)

                        cv2.putText(image, detection.label, \
                        (l + 2, t + 2), \
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), 1)

                        detections_out.detections.append(detection)

                except Exception as e:
                    print e

        return (image, detections_out)

    def publish(self, detections, image_outgoing):
        """
        Creates the ros messages and publishes them

        Args:
        detections (cob_perception_msgs/DetectionArray): incoming detections
        image_outgoing (sensor_msgs/Image): with face bounding boxes and labels

        """

        self.pub_det.publish(detections)
        self.pub_det_rgb.publish(image_outgoing)



    def initialize_database(self):
        """
        Reads the PNG images from ./people folder and
        creates a list of peoples

        The names of the image files are considered as their
        real names.

        For example;
        /people
          - mario.png
          - jennifer.png
          - melanie.png

        Returns:
        (tuple) (people_list, name_list) (features of people, names of people)

        """
        filenames = glob.glob(cd + '/people/*.png')

        people_list = []
        name_list = []

        for f in filenames:
            im = cv2.imread(f, 1)

            cv2.imshow("Database Image", im)

            cv2.waitKey(500)

            im = im.astype(np.uint8)

            people_list.append(fr.face_encodings(im)[0])

            name_list.append(f.split('/')[-1].split('.')[0])

            cv2.destroyAllWindows()

        return (people_list, name_list)

def main():
    """ main function
    """
    node = FaceRecognitionNode()

if __name__ == '__main__':
    main()
