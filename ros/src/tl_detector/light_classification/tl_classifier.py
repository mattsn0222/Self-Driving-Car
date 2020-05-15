from styx_msgs.msg import TrafficLight
import tensorflow as tf
from keras import backend as K
import cv2
import numpy as np
import rospy
import rospkg
from ssd_output_decoder import decode_detections

rospack = rospkg.RosPack()
GRAPH_FILE=rospack.get_path('tl_detector')+'/my_model4.pb'
rospy.loginfo("model path:"+GRAPH_FILE)

SSD7_GRAPH_FILE=rospack.get_path('tl_detector')+'/my_ssd7_model.pb'

inference_map_text=["Red", "Yellow", "Green", "none"]
inference_map_code=[TrafficLight.RED, TrafficLight.YELLOW, TrafficLight.GREEN, TrafficLight.UNKNOWN]

inference_map_text_ssd7=["Yellow", "Red", "Green", "none"]
inference_map_code_ssd7=[TrafficLight.YELLOW, TrafficLight.RED, TrafficLight.GREEN, TrafficLight.UNKNOWN]

class RunMode():
    NORMAL = 1
    ROSBAG_PLAYBACK = 2

RUN_MODE = RunMode.NORMAL

hackhack = 0

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
        self.using_ssd7 = True
        if RUN_MODE == RunMode.ROSBAG_PLAYBACK:
            self.is_site = True
        else:
            self.is_site = is_site

        if (self.using_ssd7):
            detection_graph = load_graph(SSD7_GRAPH_FILE)

            self.input_tensor = detection_graph.get_tensor_by_name('input_1:0')

            self.keras_learning = detection_graph.get_tensor_by_name('keras_learning_phase:0')

            self.detection = detection_graph.get_tensor_by_name('predictions/concat:0')

            self.tf_session = tf.Session(graph=detection_graph)
        #TODO : remove this inferior approach (who would even think this would work?!)
        else:
            detection_graph = load_graph(GRAPH_FILE)

            # The input placeholder for the image.
            self.input_tensor = detection_graph.get_tensor_by_name('input_tensor:0')

            # The output of the inference
            self.result_tensor = detection_graph.get_tensor_by_name('result_tensor/Reshape:0')

            # This is needed to make the Keras graph happy
            self.keras_learning = detection_graph.get_tensor_by_name('conv1_bn/keras_learning_phase:0')

            self.tf_session = tf.Session(graph=detection_graph)

    def get_classification(self, cv_image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        if (self.using_ssd7):
            # for the site image the camera is positioned poorly and a large part of the image
            # is the hood. Therefore we crop off the top left corner, which is the relevant part
            # (as long as the car is going clockwise - otherwise we should crop both the
            # top left and top right and run a recognition on both, adding the results)
            if self.is_site:
                cv_image = cv_image[0:416, 0:554].copy()
            cv_image = cv2.resize(cv_image, (416, 416))
            cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            cv_images = np.reshape(cv_image_rgb, (1, 416, 416, 3))
            y_pred = self.tf_session.run(self.detection,
                                         feed_dict={self.input_tensor: cv_images, self.keras_learning: 0})
            y_pred_decoded = decode_detections(y_pred,
                                               confidence_thresh=0.7,
                                               iou_threshold=0.45,
                                               top_k=200,
                                               normalize_coords=True,
                                               img_height=cv_image_rgb.shape[0],
                                               img_width=cv_image_rgb.shape[1])

            votes=[0]*3
            argm = 3 # default is none
            if len(y_pred_decoded) > 0:
                y_pred_list = y_pred_decoded[0].tolist()
                for cls, conf, xmin, ymin, xmax, ymax in y_pred_list:
                    votes[int(cls-1)] += 1
            if np.max(votes) != 0:
                argm = np.argmax(votes)

            rospy.logwarn('Guessed ' + inference_map_text_ssd7[argm] + " votes(YRG) " + str(votes))

            # --- debug
            #global hackhack
            #boxColors = [(255, 255, 0), (255, 0, 0), (0, 255, 0)]
            #if len(y_pred_decoded) > 0:
            #   y_pred_list = y_pred_decoded[0].tolist()
            #   for sss in y_pred_list:
            #       # print sss
            #       cls, conf, xmin, ymin, xmax, ymax = sss
            #       boxColor = boxColors[int(cls) - 1]
            #       cv2.rectangle(cv_image_rgb, (int(xmin), int(ymin)),
            #                     (int(xmax), int(ymax)), boxColor, 2)
            #cv_image_rgb = cv2.putText(cv_image_rgb,
            #                           'Guessed ' + inference_map_text_ssd7[argm] + str(votes),
            #                           (50,50), cv2.FONT_HERSHEY_SIMPLEX,
            #                           fontScale=0.5, color=(255,255,255), thickness=1)
            #outimg = cv2.cvtColor(cv_image_rgb, cv2.COLOR_RGB2BGR)
            #cv2.imwrite("recog_%04d.png" % hackhack, outimg)
            #hackhack += 1
            # ----

            return inference_map_code_ssd7[argm]
        else:
            #TODO: remove obsolete approach
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





