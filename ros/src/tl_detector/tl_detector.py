#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane, Waypoint
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
from scipy import spatial
import tf
import cv2
import yaml


STATE_COUNT_THRESHOLD = 3
LOGGING_RATE = 5  # Only log at this rate (1 / Hz)
SIMULATED_LIGHTS=False

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
            
        self.camera_image = None
        self.camera_image_is_raw = False
        self.has_image_color = False
        self.lights = []
        self.waypoints_2d = None
        self.waypoint_tree = None

        self.bridge = CvBridge()


        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        self.log_count = 0

        # load the x,y coordinates of each traffic light from config
        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        running_on_site = self.config['is_site']
        self.light_classifier = TLClassifier(running_on_site)

        # all initializations should happen before the subscriptions, otherwise the callbacks are
        # hitting uninitialized / missing members


        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb_color)
        sub7 = rospy.Subscriber('/image_raw', Image, self.image_cb_raw)


        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)


        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            # Build a KD Tree to speed up searching for waypoints in the future
            self.waypoint_tree = spatial.KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        #rospy.loginfo("image raw %d seq %d", self.camera_image_is_raw, self.camera_image.header.seq)

        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def image_cb_color(self, msg):
        self.has_image_color = True
        self.camera_image_is_raw = False
        self.image_cb(msg)

    def image_cb_raw(self, msg):
        if self.has_image_color == False:
            self.camera_image_is_raw = True
            self.image_cb(msg)

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.base_waypoints

        """
        #TODO implement
        if self.waypoint_tree is None:
            return -1
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]
        return closest_idx

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # For testing, just return the light state from simulator
        if (SIMULATED_LIGHTS):
            return light.state
        
        if(not self.has_image):
            self.prev_light_loc = None
            return TrafficLight.UNKNOWN

        cv_image = None
        if (self.camera_image_is_raw==False):
            cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        else:
            cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bayer_grbg8")
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BayerGB2BGR) # cv2 name GB is from "bayer_grbg"[-1:-2]
            #rospy.loginfo("raw image seq %d size %d,%d", self.camera_image.header.seq, cv_image.shape[0], cv_image.shape[1])

        #Get classification
        result = self.light_classifier.get_classification(cv_image)
        # write annotated image for testing
        write_annotated_image=False
        if (self.camera_image_is_raw==True and write_annotated_image==True):
            color=(0,0,0)
            if result == TrafficLight.GREEN:
                color=(0, 255, 0)
            elif result == TrafficLight.YELLOW:
                color=(255, 255, 0)
            elif result == TrafficLight.RED:
                color=(255, 0, 0)
            dbgimg = cv2.circle(cv_image, (20,20), 10, color,-1)
            cv2.imwrite('/home/student/shared/ros_image/raw_%04d.png'%self.camera_image.header.seq, dbgimg) #, cv2.cvtColor(dbgimg, cv2.COLOR_RGB2BGR))

        return result

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        closest_light = None
        stop_line_wp_idx = -1
        state = TrafficLight.UNKNOWN

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_position = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)
            if car_position != -1:
                #TODO find the closest visible traffic light (if one exists)
                diff = len(self.base_waypoints.waypoints)
                for i, light in enumerate(self.lights):
                    # Get stop line waypoint index
                    line = stop_line_positions[i]
                    temp_wp_idx = self.get_closest_waypoint(line[0], line[1])
                    # Find closest stop line waypoint index
                    d = temp_wp_idx - car_position
                    if d >= 0 and d < diff:
                        diff = d
                        closest_light = light
                        stop_line_wp_idx = temp_wp_idx

        if closest_light:
            self.log_count += 1
            state = self.get_light_state(closest_light)
            if (self.log_count % LOGGING_RATE) == 0:
                rospy.logwarn("DETECT: stop_line_wp_idx={}, state={}, car_position={}".format(stop_line_wp_idx, self.state_to_string(state), car_position))

        #self.base_waypoints = None
        return stop_line_wp_idx, state
    
    def state_to_string(self, state):
        out = "unknown"
        if state == TrafficLight.GREEN:
            out = "green"
        elif state == TrafficLight.YELLOW:
            out = "yellow"
        elif state == TrafficLight.RED:
            out = "red"
        return out
    
if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')

