#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from math import atan2, sqrt, degrees, asin, tan
from datetime import datetime
from os import mkdir
import random
import tf
import cv2
import yaml
import tensorflow as tf
import numpy as np
import time
from scipy.stats import norm
from scipy import spatial
from keras import backend as K
from keras.applications.mobilenet import preprocess_input

SAVING_MODE = False

SSD_GRAPH_FILE='ssd_mobilenet/frozen_inference_graph.pb'
CUSTOM_GRAPH_FILE='/home/student/GitHub/Self-Driving-Car/trainer/model/my_model4.pb'

inference_map=["Red ", "Yellow ", "Green ", "none "]


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def to_image_coords (boxes, height, width):
    box_coords=np.zeros_like(boxes)
    box_coords[:, 0] = boxes[:, 0] *  height
    box_coords[:, 1] = boxes[:, 1] * width
    box_coords[:, 2] = boxes[:, 2] * height
    box_coords[:, 3] = boxes[:, 3] * width

    return box_coords

def quaternion_to_euler(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = degrees(atan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = degrees(asin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = degrees(atan2(t3, t4))

    return X, Y, Z

def load_graph(graph_file):
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_file, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph

STATE_COUNT_THRESHOLD = 3

class TLCapture(object):
    def __init__(self):

        self.init_tensorflow_custom()
        rospy.init_node('trafficlight_capture')

        self.pose = None
        self.camera_image = None
        self.lights = []
        if SAVING_MODE:
            self.sample_dir = "data/"+ datetime.now().strftime("%m%d-%H%M%S/")
            #make one directory per traffic light state
            mkdir(self.sample_dir)
            for ds in ["training", "validation"]:
                mkdir(self.sample_dir + "/" + ds)
                for state in ['0', '1', '2', '3']:
                    mkdir(self.sample_dir + "/" + ds + "/" + state)

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        # load the x,y coordinates of each traffic light from config
        config_string = rospy.get_param("/traffic_light_config")
        print config_string
        self.config = yaml.load(config_string)

        self.image_pub = rospy.Publisher('/image_recognized', Image, queue_size=1)

        self.bridge = CvBridge()

        running_on_site = self.config['is_site']
        if running_on_site:
            sub7 = rospy.Subscriber('/camera_info', Image, self.camera_info)

        rospy.spin()

    def init_tensorflow_custom(self):
        detection_graph = load_graph(CUSTOM_GRAPH_FILE)

        # The input placeholder for the image.
        self.input_tensor = detection_graph.get_tensor_by_name('input_tensor:0')

        # The output of the inference
        self.result_tensor = detection_graph.get_tensor_by_name('result_tensor/Reshape:0')

        self.keras_learning = detection_graph.get_tensor_by_name('conv1_bn/keras_learning_phase:0')

        self.tf_session = tf.Session(graph=detection_graph)


    def init_tensorflow_sdd(self):
        detection_graph = load_graph(SSD_GRAPH_FILE)

        # The input placeholder for the image.
        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')

        # The classification of the object (integer id).
        self.detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

        self.tf_session = tf.Session(graph=detection_graph)

    def save_image(self, msg, state_name):
        dset_name = "training"
        if (random.random() < 0.2):
            dset_name = "validation"
        cv_image_bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        image_fname = datetime.now().strftime("%m%d-%H%M%S") + str(self.pose.header.seq) + ".png"
        image_dirname = self.sample_dir + "/" + dset_name + "/" + state_name + "/"
        cv2.imwrite(image_dirname + image_fname, cv_image_bgr)
        print "saved ", image_fname

    def show_distance(self):
        for light in self.lights:
            lightx = light.pose.pose.position.x
            lighty = light.pose.pose.position.y
            carx = self.pose.pose.position.x
            cary = self.pose.pose.position.y
            dx = lightx-carx
            dy = lighty-cary
            dist = sqrt(dx*dx+dy*dy)
            if (dist < 250 and dist > 5):
                sign_relative_angle = np.rad2deg(atan2(dy, dx))
                Xa, Ya, Za = quaternion_to_euler(self.pose.pose.orientation.x, self.pose.pose.orientation.y,
                                                 self.pose.pose.orientation.z, self.pose.pose.orientation.w)

                angle_difference = ((sign_relative_angle - Za + 180) % 360) - 180
                if (abs(angle_difference) < 15):
                    print "dist: ", dist, "angle", angle_difference

    def save_sample_image(self, msg):
        may_be_visible = False
        for light in self.lights:
            lightx = light.pose.pose.position.x
            lighty = light.pose.pose.position.y
            lightz = light.pose.pose.position.z
            carx = self.pose.pose.position.x
            cary = self.pose.pose.position.y
            carz = self.pose.pose.position.z
            dx = lightx-carx
            dy = lighty-cary
            dist = sqrt(dx*dx+dy*dy)
            if (dist < 200 and dist > 5):
                print "dist: ", dist
                sign_relative_angle = np.rad2deg(atan2(dy, dx))
                dz = lightz - carz
                Xa, Ya, Za = quaternion_to_euler(self.pose.pose.orientation.x, self.pose.pose.orientation.y,
                                                 self.pose.pose.orientation.z, self.pose.pose.orientation.w)

                #print "l: ", lightx, lighty, "c: ", carx, cary
                #print "D:", dist, "dz:", dz, "cw", carw, "ls:",light.state
                #print "X Y X", X, Y, Z
                #print "cam rel", sign_relative_angle
                angle_difference = ((sign_relative_angle - Za + 180) % 360) - 180
                angle_of_visibility  = np.rad2deg(atan2(3.5,dist)) # calculate the angle at which at least one of the three lights are visible
                angle_treshold = 8.6+angle_of_visibility # viewing angle of camera
                #print "D:", dist, "ad", angle_difference, "at", angle_treshold


                if abs(angle_difference) < angle_treshold:
                    if dist < 150 and dist > 21:
                        self.save_image(msg, str(light.state))
                    else:
                        may_be_visible = True
        #also record "nothing" states
        if may_be_visible == False:
            if (random.random() < 0.1):
                self.save_image(msg, "3")

    def mobilenet_preprocess(self, cv_image):
        x = cv_image/128.
        x -= 0.5
        return x

    def perform_detect_custom(self, cv_image):
        cv_image=cv2.resize(cv_image, (224,224))
        #cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        preprocessed_input = self.mobilenet_preprocess(cv_image)
        #TODO: is a BGR2RGB conversion needed?
        inference_result = self.tf_session.run(self.result_tensor,
                            feed_dict={self.input_tensor: (preprocessed_input,), self.keras_learning: 0})

        best_guess = np.argmax(inference_result)
        confidence = softmax(inference_result[0])

        cv2.putText(cv_image, inference_map[best_guess] + str(confidence[best_guess]),  (0, 150), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color=(255,255,255))

        return cv_image

    def perform_detect(self, cv_image):
        (boxes, scores, classes) = self.tf_session.run([self.detection_boxes, self.detection_scores, self.detection_classes],
                                            feed_dict={self.image_tensor: (cv_image, )})

        # Remove unnecessary dimensions
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        n = len(classes)
        min_score = 0.8
        idxs=[]
        for i in range(n):
            if scores[i] >= min_score:
                idxs.append(i)

        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]

        height, width, _ = cv_image.shape
        box_coords = to_image_coords(filtered_boxes, height, width)

        self.draw_boxes(cv_image, box_coords, filtered_classes)

        return cv_image


    def draw_boxes(self, cv_image, box_coords, classes):
        for box, cls in zip(box_coords, classes):
            color = (255,255,255)
            if cls == 8:
                color = (0,0,255)
            cv2.rectangle(cv_image, (box[1],box[0]), (box[3], box[2]), color=color, thickness=2)

    def pose_cb(self, msg):
        self.pose = msg

    def traffic_cb(self, msg):
        self.lights = msg.lights
        #print "Got traffic lights ", len(msg.lights), self.lights[0]


    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """

        if SAVING_MODE:
            # TODO: undistort the image if it is coming from the real car
            self.save_sample_image(msg)

        else:
            self.show_distance()
            cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            rospy.logwarn("got image")
            #Get classification
            cv_image = self.perform_detect_custom(cv_image)

            final_img = self.bridge.cv2_to_imgmsg(cv_image, "rgb8")

            self.image_pub.publish(final_img)



    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        #if(self.pose):
        #    car_position = self.get_closest_waypoint(self.pose.pose)

        #TODO find the closest visible traffic light (if one exists)

        if light:
            state = self.get_light_state(light)
            return light_wp, state
        self.waypoints = None
        return -1, TrafficLight.UNKNOWN

    def camera_info(self, msg):
        #TODO: get distortion parameters
        pass

if __name__ == '__main__':
    try:
        TLCapture()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
