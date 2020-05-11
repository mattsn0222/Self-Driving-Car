#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
import numpy as np
from scipy import spatial
import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
MAX_JERK = 0.5 # m/s2
MAX_ACCEL = 0.5 # m/s2
PUBLISH_RATE = 30 # the consumer (waypoint follower) is running at 30 Hz supposedly

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        #rospy.Subscriber('/obstacle_waypoint', Int32, self.obstacle_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.pose = None
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.stopline_wp_idx = -1

        self.loop()
    
    def loop(self):
        rate = rospy.Rate(PUBLISH_RATE)
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints:
                self.publish_waypoints()
            rate.sleep()
    
    def get_closest_waypoint_idx(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        closest_idx = self.waypoint_tree.query([x,y], 1)[1]
        
        # Check if closest is ahead or behind vehicle
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx-1]
        
        # Equation for hyperplane through closest_coords
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x, y])

        # take the dot product of the vector from the previous waypoint to the closest waypoint
        # and the vector from the closes waypoint to the car. If it is positive (the two vectors
        # point in the same direction), the car is past the current (closest) waypoint,
        # otherwise it is behind the current waypoint (heading for the closest waypoint)
        val = np.dot(cl_vect-prev_vect, cl_vect-pos_vect)
        
        if val < 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
        return closest_idx
    
    def publish_waypoints(self):
        final_lane = self.generate_trajectory()
        self.final_waypoints_pub.publish(final_lane)
    
    def generate_trajectory(self):
        lane = Lane()
        #lane.header = self.base_waypoints.header
        closest_idx = self.get_closest_waypoint_idx()
        last_idx = closest_idx + LOOKAHEAD_WPS
        # Get the next section of waypoints
        base_waypoints = self.base_waypoints.waypoints[closest_idx:last_idx]
        
        # No changes to trajectory
        if (self.stopline_wp_idx == -1) or (self.stopline_wp_idx >= last_idx):
            lane.waypoints = base_waypoints
        # Action needed
        else:
            lane.waypoints = self.brake_action(base_waypoints, closest_idx)
        
        return lane
        
    def brake_action(self, waypoints, closest_idx):
        # Array for brake waypoints to later be merged in
        brake = []
        # Enumerate over the list of waypoints
        for i, wp in enumerate(waypoints):
            # Add new Waypoint object
            o = Waypoint()
            o.pose = wp.pose
            # Center of car
            stop_idx = max(self.stopline_wp_idx - closest_idx - 2, 0)
            # Calculate distance to start to decelerate
            dist = self.distance(waypoints, i, stop_idx)
            vel = math.sqrt(2*MAX_JERK*dist)
            
            # If the car is barely moving stop the car
            if vel < 1.0:
                vel = 0.0
            
            # To reduce large square roots when the distance is far
            o.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            brake.append(o)
        
        return brake

    def pose_cb(self, msg):
        # TODO: Implement
        #pass
        self.pose = msg

    def waypoints_cb(self, waypoints):
        # TODO: Implement
        #save the base waypoints. This callback is supposed to be called only once
        self.base_waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            # build a KD Tree to speed up searching for waypoints in the future
            self.waypoint_tree = spatial.KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        # you get the index of the waypoint that is closest to an upcoming red light, e.g. 12 for waypoints[12]
        # Target velocity should be set to 0 at this point so the car can smoothly stop there
        # pass
        self.stopline_wp_idx = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
