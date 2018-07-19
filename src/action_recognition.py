#!/usr/bin/env python
"""
A ROS node to get action recognition results by using i3d model in tensorflow
hub. All the code is based on example colab provided by tensorflow team. Here
you can find the link to notebook:
https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/action_recognition_with_tf_hub.ipynb

Author:
    Cagatay Odabasi -- cagatay.odabasi@ipa.fraunhofer.de
"""

import rospy

import numpy as np

from cob_perception_msgs.msg import ActionRecognitionmsg

import cv2

from cv_bridge import CvBridge

from sensor_msgs.msg import Image

import rospkg

from six.moves import urllib

import tensorflow as tf

import tensorflow_hub as hub

# Get the package directory
rospack = rospkg.RosPack()

cd = rospack.get_path('cob_people_object_detection_tensorflow')


class ActionRecognitionNode(object):
    """A ROS node to get action recognition prediction by using models in
    tensorflow hub

    """
    def __init__(self):
        super(ActionRecognitionNode, self).__init__()

        # init the node
        rospy.init_node('action_recognition_node', anonymous=False)

        # Get the parameters
        (image_topic, output_topic, recognition_interval,
            buffer_size) \
            = self.get_parameters()

        self._bridge = CvBridge()

        # Advertise the result of Action Recognition
        self.pub_rec = rospy.Publisher(output_topic, \
            ActionRecognitionmsg, queue_size=1)

        # This amount of break will be given between each prediction
        self.recognition_interval = recognition_interval

        # Counter for number of frames
        self.frame_count = 0

        # frame buffer size
        self.buffer_size = buffer_size

        # Frame buffer
        self.frame_buffer = []

        # Get the labels
        self.labels = self.initialize_database()

        # Create the graph object
        self._graph = tf.Graph()

        # Initialite the model
        self.initialize_model()

        self.sub_image =rospy.Subscriber(image_topic, Image, self.rgb_callback)

        # spin
        rospy.spin()

    def get_parameters(self):
        """
        Gets the necessary parameters from parameter server

        Args:

        Returns:
        (tuple) (camera_topic,  output_topic, recognition_interval,
        buffer_size)

        """

        camera_topic = rospy.get_param("~camera_topic")
        output_topic = rospy.get_param("~output_topic")
        recognition_interval = rospy.get_param("~recognition_interval")
        buffer_size = rospy.get_param("~buffer_size")

        return (camera_topic, output_topic,
            recognition_interval, buffer_size)


    def shutdown(self):
        """
        Shuts down the node
        """
        rospy.signal_shutdown("See ya!")

    def rgb_callback(self, image):
        """
        Callback for RGB images

        Args:
        image (sensor_msgs/Image): RGB image from camera

        """

        cv_rgb = self._bridge.imgmsg_to_cv2(image, "passthrough")[:, :, ::-1]

        cv_rgb = cv2.resize(cv_rgb, (224, 224))

        cv_rgb = cv_rgb.astype(np.uint8)

        cv_rgb = cv_rgb[:, :, [2, 1, 0]]

        cv_rgb = cv_rgb/255

        # Add new frame to buffer
        self.frame_buffer.append(cv_rgb)

        if self.frame_count > self.buffer_size:
            self.frame_buffer.pop(0)

            if self.frame_count%self.recognition_interval == 0:
                print(self.frame_count)
                print(self.recognition_interval)
                # Perform action recognition
                probs = self.recognize()

                self.publish(image, probs)

        self.frame_count += 1


    def recognize(self):
        """
        Recognize the actions performed in the scene

        Args:
        (numpy.ndarray) image: incoming image

        Returns:

        (numpy.array) probabilities returned by i3d model

        """

        frames = np.asarray(self.frame_buffer)

        print(frames.shape)

        model_input = np.expand_dims(frames, axis=0)

        # Perform the inference
        with self._graph.as_default():
            with tf.train.MonitoredSession() as session:
                [ps] = session.run(self.probabilities,
                       feed_dict={self.input_placeholder: model_input})


        print("Top 5 actions:")
        for i in np.argsort(ps)[::-1][:5]:
            print("%-22s %.2f%%" % (self.labels[i], ps[i] * 100))

        return ps

    def publish(self, image, probs):
        """
        Creates the ros messages and publishes them

        Args:
        detections (cob_perception_msgs/DetectionArray): incoming detections
        """

        msg = ActionRecognitionmsg()

        msg.header = image.header

        msg.probabilities = probs.tolist()

        msg.labels = self.labels

        self.pub_rec.publish(msg)

    def initialize_database(self):
        """
        Reads the action labels from the text file

        The names of the image files are considered as their

        Returns:
        (list) labels

        """

        labels = None

        with open(cd + "/action_recognition/labels.txt", "rb") as f:
            labels = [line.decode("utf-8").strip() for line in f.readlines()]

        return labels

    def initialize_model(self):
        """
        Downloads the model from tensorflow hub and create the model object for
        inference

        """
        with self._graph.as_default():
            self._module = hub.Module("http://tfhub.dev/deepmind/i3d-kinetics-600/1")
            self.input_placeholder = tf.placeholder(shape=(None, None, 224, 224, 3), dtype=tf.float32)
            logits = self._module(self.input_placeholder)
            self.probabilities = tf.nn.softmax(logits)

def main():
    """ main function
    """
    node = ActionRecognitionNode()

if __name__ == '__main__':
    main()
