#!/bin/bash
#Script to change directories and run scripts

cd /home/workspace/CarND-Capstone

cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch

"""
# Clean
echo Starting to clean ...
./clean.sh
echo Done!

# Build
echo Starting to build ...
./build.sh

cd build
if [[ -f pid ]]
then
    echo Done!
    cd /home/workspace/CarND-PID-Control-Project
    # Run
    echo Starting to run ...
    ./run.sh
fi
"""