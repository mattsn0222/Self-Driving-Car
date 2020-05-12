from styx_msgs.msg import TrafficLight
import tensorflow as tf
from keras import backend as K
import cv2
import numpy as np
import rospy
import rospkg

rospack = rospkg.RosPack()
GRAPH_FILE=rospack.get_path('tl_detector')+'/my_model4.pb'
rospy.loginfo("model path:"+GRAPH_FILE)

#GRAPH_FILE='/home/student/GitHub/Self-Driving-Car/trainer/model/my_model4.pb'
# alternative graph with no "None" state
#GRAPH_FILE='/home/student/GitHub/Self-Driving-Car/trainer/model/my_model3.pb'

inference_map_text=["Red", "Yellow", "Green", "none"]
inference_map_code=[TrafficLight.RED, TrafficLight.YELLOW, TrafficLight.GREEN, TrafficLight.UNKNOWN]

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def load_graph(graph_file):
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_file, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph

class TLClassifier(object):
    def __init__(self, is_site):
        # maybe load different classifier for site and for simulator
        #TODO make site-specific classifier

        detection_graph = load_graph(GRAPH_FILE)

        # The input placeholder for the image.
        self.input_tensor = detection_graph.get_tensor_by_name('input_tensor:0')

        # The output of the inference
        self.result_tensor = detection_graph.get_tensor_by_name('result_tensor/Reshape:0')

        # This is needed to make the Keras graph happy
        self.keras_learning = detection_graph.get_tensor_by_name('conv1_bn/keras_learning_phase:0')

        self.tf_session = tf.Session(graph=detection_graph)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        cv_image = cv2.resize(image, (224, 224))
        preprocessed_input = self.mobilenet_preprocess(cv_image)

        inference_result = self.tf_session.run(self.result_tensor,
                                               feed_dict={self.input_tensor: (preprocessed_input,),
                                                          self.keras_learning: 0})
        best_guess = np.argmax(inference_result)
        #rospy.logwarn('Guessed ' + inference_map_text[best_guess])
        #confidence = softmax(inference_result[0])

        return inference_map_code[best_guess]

    def mobilenet_preprocess(self, cv_image):
        x = cv_image/128.
        x -= 0.5
        return x





