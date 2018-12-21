import cv2
import numpy as np
from keras import backend as k
from Vgg16 import *
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import RMSprop, SGD, Adam
from keras.initializers import VarianceScaling
import random


def get_reward_movement(iou, new_iou):
    if new_iou > iou:
        reward = 1
    else:
        reward = - 1
    return reward


def get_reward_trigger(new_iou):
    if new_iou > iou_threshold:
        reward = 3
    else:
        reward = - 3
    return reward


def get_state(image, history_vector, model):

    image_feature = feature_output(image, model)
    image_feature = np.reshape(image_feature, (visual_descriptor_size, 1))

    history_vector = np.reshape(history_vector, (number_of_actions * actions_of_history, 1))
    state = np.vstack((image_feature, history_vector))

    return state


def update_history_vector(history_vector, action):

    # Do not calculate the start action
    action_vector = np.zeros(number_of_actions)
    action_vector[action-1] = 1

    size_history_vector = np.size(np.nonzero(history_vector))
    updated_history_vector = np.zeros(number_of_actions*actions_of_history)
    if size_history_vector < actions_of_history:
        aux2 = 0
        for l in range(number_of_actions*size_history_vector,
                       number_of_actions*size_history_vector+number_of_actions - 1):

            history_vector[l] = action_vector[aux2]
            aux2 += 1
        return history_vector

    else:
        for j in range(0, number_of_actions*(actions_of_history-1) - 1):
            updated_history_vector[j] = history_vector[j+number_of_actions]
        aux = 0
        for k in range(number_of_actions*(actions_of_history-1), number_of_actions*actions_of_history):
            updated_history_vector[k] = action_vector[aux]
            aux += 1
        return updated_history_vector


def q_network(weights_path="./../../models_image_zooms/model3.h5"):
    rl_model = Sequential()
    rl_model.add(Dense(1024, activation='relu', kernel_initializer=lambda shape: VarianceScaling(scale=0.01)(shape), input_shape=(25112,)))
    rl_model.add(Dropout(0.2))
    rl_model.add(Dense(1024, activation='relu', kernel_initializer=lambda shape: VarianceScaling(scale=0.01)(shape)))
    rl_model.add(Dropout(0.2))
    rl_model.add(Dense(6, activation='linear', kernel_initializer=lambda shape: VarianceScaling(scale=0.01)(shape)))
    adam = Adam(lr=1e-6)
    rl_model.compile(loss='mse', optimizer=adam)

    if not weights_path == '0':
        rl_model.load_weights(weights_path)
        print('Read')

    return rl_model


# def qn_pascal(weights_path, object_id):
#     q_networks = []
#     if weights_path == "0":
#         # 20 for 20 classes
#         for i in range(20):
#             q_networks.append(q_network("0"))
#     else:
#         for i in range(20):
#             if i == (object_id - 1):
#                 q_networks.append(q_network(weights_path + "/model" + str(object_id) + ".h5"))
#             else:
#                 q_networks.append(q_network("0"))
#     return np.array([q_networks])
