#!/bin/bash
#Script to make, source, and launch

pip install -r requirements.txt

cd ..
sudo apt-get update
sudo apt-get install -y ros-kinetic-dbw-mkz-msgs
cd /home/workspace/Self-Driving-Car/ros
rosdep install --from-paths src --ignore-src --rosdistro=kinetic -y

cd /home/workspace/Self-Driving-Car/ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
