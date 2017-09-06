# cob_people_object_detection_tensorflow

ROS(Robot Operating System) port of Tensorflow Object Detection API

The codes are based on jupyter notebook inside of the object detection API.

Sample frame:
![alt text](https://raw.githubusercontent.com/cagbal/cob_people_object_detection_tensorflow/master/images/screenshot.png)

# Features

  - Detects the objects in images coming from a camera topic  
  - Publishes the scores, bounding boxes and labes of detection
  - Publishes detection image with bounding boxes as a sensor_msgs/Image
  - Parameters can be set fom a Yaml file

### Tech

This repo uses a number of open source projects to work properly:

* [Tensorflow]
* [Tensorflow-Object Detection API]
* [ROS]
* [Numpy]

### Installation

First, tensorflow should be installed on your system.

Then,
```sh
$ cd && mkdir -p catkin_ws/src && cd ..
$ catkin_make && cd src
$ git clone https://github.com/cagbal/cob_people_object_detection_tensorflow.git
$ git clone https://github.com/ipa-rmb/cob_perception_common.git
$ cd cob_people_object_detection_tensorflow/src
$ protoc object_detection/protos/*.proto --python_out=.
$ cd ~/catkin_ws
$ rosdep install --from-path src/ -y -i
$ catkin_make
```

### Running

Turn on your camera driver in ROS and set your input RGB topic name in yaml config file under launch directory. The default is for openni2.

Then,

```sh
$ roslaunch cob_people_object_detection_tensorflow cob_people_object_detection_tensorflow.launch 
```

##### Subscibes to:
- To any RGB image topic that you set in *params.yaml file. 

##### Publishes to:
- /object_detection/detections (cob_perception_msgs/DetectionArray) Includes all the detections with probabilities, labels and bounding boxes
- /object_detection/detections_image (sensor_msgs/Image) The image with bounding boxes 

### Performance 
The five last detection times from my computer(Intel(R) Core(TM) i7-6820HK CPU @ 2.70GHz) in seconds: 
0.105810880661
0.108750104904
0.112195014954
0.115020036697
0.108013153076

License
----

Apache (but please also look at tensorflow and tf object detection licences)
