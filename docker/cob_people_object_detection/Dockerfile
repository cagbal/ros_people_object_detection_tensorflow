ARG ROS_VERSION=noetic

FROM ros:${ROS_VERSION}

ARG ROS_VERSION

LABEL maintainer "cagatay.odabasi@ipa.fraunhofer.de"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y libopencv-dev \
    git \
    protobuf-compiler \
    python3-pip 

RUN mkdir -p /root/catkin_ws/src

# Install the package 
WORKDIR /root/catkin_ws/src

RUN git clone --recursive https://github.com/cagbal/cob_people_object_detection_tensorflow.git

WORKDIR /root/catkin_ws/src/cob_people_object_detection_tensorflow/src

RUN protoc object_detection/protos/anchor_generator.proto --python_out=.

WORKDIR /root/catkin_ws

#RUN apt-key update && apt-get update && rosdep install --from-path /root/catkin_ws/src/ -y -i 


# 
RUN apt-get update && apt-get install -y --allow-unauthenticated \
     ros-${ROS_VERSION}-cv-bridge \
     ros-${ROS_VERSION}-eigen-conversions \
     ros-${ROS_VERSION}-image-transport \
     ros-${ROS_VERSION}-pcl-ros \
     ros-${ROS_VERSION}-cob-perception-common

RUN apt-get update && apt-get install -y libtinyxml-dev

# Build 
RUN /bin/bash -c 'source /opt/ros/${ROS_VERSION}/setup.bash && catkin_make'

# Install packages with pip
RUN pip3 install --upgrade --force-reinstall pip==20.03

RUN pip3 install tensorflow \
    scipy \
    filterpy \ 
    matplotlib \
    face_recognition \ 
    sklearn \
    numba

