#!/bin/bash
#Script to change directories and run scripts

cd /home/workspace/Self-Driving-Car

cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
