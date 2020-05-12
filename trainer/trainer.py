from keras.layers import Dense, GlobalAveragePooling2D, Reshape, Dropout, Conv2D, Activation
from keras.models import Model
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import preprocess_input
from keras.layers import Input, Lambda
import tensorflow as tf
from keras.applications.mobilenet import MobileNet
from keras import backend as K

# I couldn't get the freeze_graph.py script working, so I took this approach to generate the pb file from
# https://stackoverflow.com/questions/45466020/how-to-export-keras-h5-to-tensorflow-pb
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

#option to include "none" state or exclude it
#CLASSES_TO_TRAIN=["0", "1", "2", "3"]
CLASSES_TO_TRAIN=["0", "1", "2"]
NUMBER_OF_CLASSES=len(CLASSES_TO_TRAIN)
EPOCHS=10

freeze_flag = False  # With frozen layers the results were much worse, so I let Keras retrain the whole model
weights_flag = 'imagenet' # 'imagenet' or None
preprocess_flag = True # Should be true for ImageNet pre-trained typically

# This is the largest supported size for the Keras application
input_size = (224,224,3)

# we have to remove the top to change the number of classes
mobilenet = MobileNet(input_shape=input_size, include_top=False,
    weights=weights_flag)

if freeze_flag == True:
    for layer in mobilenet.layers:
        layer.trainable = False


# Makes the input placeholder layer 32x32x3 for CIFAR-10
input_ph = Input(shape=(224,224,3), name = "input_tensor")

inp = mobilenet(input_ph)

# Add back the missing top (based on the mobilenet.py in Keras)
gap = GlobalAveragePooling2D() (inp)
gap = Reshape((1,1,1024)) (gap)
gap = Dropout(1e-3) (gap)
gap = Conv2D (NUMBER_OF_CLASSES, (1,1), padding='same', name='conv_preds') (gap)
gap = Activation ('softmax', name='act_softmax') (gap)
predictions = Reshape ((NUMBER_OF_CLASSES,), name='result_tensor') (gap)

model = Model(inputs=input_ph, outputs=predictions)

# Compile the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Check the summary of this new model to confirm the architecture
model.summary()


BATCH_SIZE = 32

datagen_train = ImageDataGenerator(preprocessing_function=preprocess_input, 
                                   rotation_range = 5, width_shift_range = 0.2, 
                                   height_shift_range = 0.2,  shear_range=0.2, 
                                   zoom_range=0.2, horizontal_flip = True)
datagen_valid = ImageDataGenerator(preprocessing_function=preprocess_input)


train_gen = datagen_train.flow_from_directory(directory="data/training",
    target_size=(224, 224),
    color_mode="rgb",
    class_mode="categorical",
    classes=CLASSES_TO_TRAIN,
    batch_size=BATCH_SIZE,
    shuffle=True
)


valid_gen = datagen_valid.flow_from_directory(directory="data/validation",
    target_size=(224, 224),
    color_mode="rgb",
    class_mode="categorical",
    classes=CLASSES_TO_TRAIN,
    batch_size=BATCH_SIZE,
    shuffle=True
)



model.fit_generator(train_gen, 
                    steps_per_epoch = train_gen.samples // BATCH_SIZE,
                    validation_data = valid_gen, 
                    validation_steps = valid_gen.samples // BATCH_SIZE,
                    epochs = EPOCHS, verbose=1)

print(model.output.op.name)
frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in model.outputs])

tf.train.write_graph(frozen_graph, "model", "my_model" + str(NUMBER_OF_CLASSES) + ".pb", as_text=False)


