# Self Driving Car Capstone Project
The goal of this project is to use Ros code to integrate with Carla, Udacity's Self Driving Car. To get there our team utilized a simulator that works very similarly to Carla to drive on a simulated highway with traffic lights. The project then transitions into the real world and is tested on Carla in a closed track to mimick various conditions similar to a real public road environment.

[//]: # (Image References)

[image1]: /img/final-project-ros-graph-v2.png "ROS System"

### Team
| Name | Email |
| :-------------------------------: |:------------------------------------:|
| Krishan Patel (Team Leader)  | kspatel95@gmail.com |
| Tamás Kerecsen                    | kerecsen@gmail.com |
| Tomoya Matsumura               | tomoya.matsumura@gmail.com |
| Gyorgy Blahut                        | fable3@gmail.com |

### To get Ros code started (Simulator)
```
1. cd Self-Driving-Car
2. ./run.sh
```
### If setup is needed for downloading required packages
``` 
./setup.sh
```
---

## Ros System

The Ros system utilized in the Simulator is intended to be modeled very similarly to the Udacity Self Driving Car. 

![Final score][image1]

### Main Components of the project
---
#### Waypoint Updater

The simulated car as well as the self driving car both have waypoints that are given and feed into the Ros System. The car is given base waypoints and saves it but only looks at a portion of it that is relevant to the car. The car begins by finding the closest waypoint ahead of it and then only handles a certain number of waypoints ahead of it. Another task that the waypoint updater handles is any adjustments needed to be made for traffic lights or obstacles. This allows for the car to know when to begin slowing down, where the stop line is, and when to search for relevant traffic lights.

#### Traffic Light Detection



#### Drive By Wire Node

The DBW Node uses the data coming from the waypoint updater to know when to apply different commands to throttle, braking, and steering. By default the car is going to throttle till it hits the speed limit or the vehicle limit, as the waypoints require some steering or braking the car relies on updates from the twist_controller.py. The Twist Controller  utilizes a combination of PID, a Low Pass Filter, and a Yaw Controller all to have a smoother and more effective response to control the car. The Twist Controller ensures that the car is able to make a full stop when needed, steer to stay within the lanes, and maintain speed when possible which feeds directly into the DBW Node when enabled.

---

### Udacity Capstone Project README

This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. For more information about the project, see the project introduction [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).

Please use **one** of the two installation options, either native **or** docker installation.

### Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases).

### Docker Installation
[Install Docker](https://docs.docker.com/engine/installation/)

Build the docker container
```bash
docker build . -t capstone
```

Run the docker file
```bash
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
```

### Port Forwarding
To set up port forwarding, please refer to the "uWebSocketIO Starter Guide" found in the classroom (see Extended Kalman Filter Project lesson).

### Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

### Real world testing
1. Download [training bag](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic_light_bag_file.zip) that was recorded on the Udacity self-driving car.
2. Unzip the file
```bash
unzip traffic_light_bag_file.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_file/traffic_light_training.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. Confirm that traffic light detection works on real life images

### Other library/driver information
Outside of `requirements.txt`, here is information on other driver/library versions used in the simulator and Carla:

Specific to these libraries, the simulator grader and Carla use the following:

|        | Simulator | Carla  |
| :-----------: |:-------------:| :-----:|
| Nvidia driver | 384.130 | 384.130 |
| CUDA | 8.0.61 | 8.0.61 |
| cuDNN | 6.0.21 | 6.0.21 |
| TensorRT | N/A | N/A |
| OpenCV | 3.2.0-dev | 2.4.8 |
| OpenMP | N/A | N/A |

We are working on a fix to line up the OpenCV versions between the two.
