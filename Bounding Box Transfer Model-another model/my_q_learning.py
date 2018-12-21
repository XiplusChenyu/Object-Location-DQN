import random
import cv2
import keras
from keras.layers import Dense, K
import tensorflow as tf
import math
import sys
from my_load_data import load_data
import numpy as np

history_size = 10
action_option = 9
max_steps = 20
experience_sample_size = 20
max_experience_size = 1000
gamma = 0.1
epsilon_change_steps = 10
loss_arr = []


def extract_feature(image, history, vgg16):
    history_feature = np.zeros(action_option * history_size)
    for i in range(history_size):
        if history[i] != -1:
            history_feature[i * action_option + history[i]] = 1

    feature_extractor = K.function([vgg16.layers[0].input], [vgg16.layers[20].output])
    image_reshape = [(cv2.resize(image, (224, 224))).reshape(1, 224, 224, 3)]
    image_feature = feature_extractor(image_reshape)[0]
    image_feature = np.ndarray.flatten(image_feature)
    feature = np.concatenate((image_feature, history_feature))

    return np.array([feature])


def compute_q(feature, deep_q_model):
    output = deep_q_model.predict(feature)
    return np.ndarray.flatten(output)


def compute_mask(action, current_mask):
    image_rate = 0.1
    delta_width = image_rate * (current_mask[2] - current_mask[0])
    delta_height = image_rate * (current_mask[3] - current_mask[1])
    dx1 = 0
    dy1 = 0
    dx2 = 0
    dy2 = 0

    if action == 0:
        dx1 = delta_width
        dx2 = delta_width
    elif action == 1:
        dx1 = -delta_width
        dx2 = -delta_width
    elif action == 2:
        dy1 = delta_height
        dy2 = delta_height
    elif action == 3:
        dy1 = -delta_height
        dy2 = -delta_height
    elif action == 4:
        dx1 = -delta_width
        dx2 = delta_width
        dy1 = -delta_height
        dy2 = delta_height
    elif action == 5:
        dx1 = delta_width
        dx2 = -delta_width
        dy1 = delta_height
        dy2 = -delta_height
    elif action == 6:
        dy1 = delta_height
        dy2 = -delta_height
    elif action == 7:
        dx1 = delta_width
        dx2 = -delta_width

    new_mask_tmp = np.array([current_mask[0] + dx1, current_mask[1] + dy1,
                         current_mask[2] + dx2, current_mask[3] + dy2])
    new_mask = np.array([
        min(new_mask_tmp[0], new_mask_tmp[2]),
        min(new_mask_tmp[1], new_mask_tmp[3]),
        max(new_mask_tmp[0], new_mask_tmp[2]),
        max(new_mask_tmp[1], new_mask_tmp[3])
    ])

    return new_mask


def compute_iou(mask, ground_truth):
    dx = min(mask[2], ground_truth[2]) - max(mask[0], ground_truth[0])
    dy = min(mask[3], ground_truth[3]) - max(mask[1], ground_truth[1])

    if (dx >= 0) and (dy >= 0):
        inter_area = dx*dy
    else:
        inter_area = 0

    mask_area = (mask[2] - mask[0]) * (mask[3] - mask[1])
    ground_truth_area = (ground_truth[2] - ground_truth[0]) * (ground_truth[3] - ground_truth[1])

    return inter_area / (mask_area + ground_truth_area - inter_area)


def compute_reward(action, ground_truth, current_mask):
    new_mask = compute_mask(action, current_mask)
    iou_new = compute_iou(new_mask, ground_truth)
    iou_current = compute_iou(current_mask, ground_truth)

    if iou_current < iou_new:
        return 1
    else:
        return -1


def compute_end_reward(current_mask, ground_truth):
    if compute_iou(current_mask, ground_truth) > 0.6:
        return 3
    else:
        return -3


def select_action(feature, ground_truth_box, step, q_value, epsilon, current_mask):
    if step == max_steps:
        action = 8

    else:
        if random.random() > epsilon:
            action = np.argmax(q_value)
        else:
            end_reward = compute_end_reward(current_mask, ground_truth_box)
            if end_reward > 0:
                action = 8
            else:
                rewards = []
                for i in range(action_option - 1):
                    reward = compute_reward(i, ground_truth_box, current_mask)
                    rewards.append(reward)
                rewards = np.asarray(rewards)
                positive_reward_index = np.where(rewards >= 0)[0]

                if len(positive_reward_index) == 0:
                    positive_reward_index = np.asarray(range(9))

                action = np.random.choice(positive_reward_index)

    return action


def execute_action(action, history, ground_truth_box, current_mask):
    if action == 8:
        new_mask = current_mask
        reward = compute_end_reward(current_mask, ground_truth_box)
        end = True
    else:
        new_mask = compute_mask(action, current_mask)
        reward = compute_reward(action, ground_truth_box, current_mask)
        history = history[1:]
        history.append(action)
        end = False

    return new_mask, reward, end, history


def compute_target(reward, new_feature, model):
    return reward + gamma * np.amax(compute_q(new_feature, model))


def crop_image(image, new_mask):
    height, width, channel = np.shape(image)
    new_mask = np.asarray(new_mask).astype("int")
    new_mask[0] = max(new_mask[0], 0)
    new_mask[1] = max(new_mask[1], 0)
    new_mask[2] = min(new_mask[2], width)
    new_mask[3] = min(new_mask[3], height)
    cropped_image = image[new_mask[1]:new_mask[3], new_mask[0]:new_mask[2]]
    new_height, new_width, new_channel = np.shape(cropped_image)

    if new_height == 0 or new_width == 0:
        cropped_image = np.zeros((224, 224, 3))
    else:
        cv2.resize(cropped_image, (224, 224))

    return cropped_image


def experience_replay(deep_q_model, experience):
    sample_index = random.choices(range(len(experience)), k=experience_sample_size)
    sample = [experience[i] for i in sample_index]

    targets = np.zeros((experience_sample_size, action_option))

    for i in range(experience_sample_size):
        feature, action, new_feature, reward, end = sample[i]
        target = reward

        if not end:
            target = compute_target(reward, new_feature, deep_q_model)

        targets[i, :] = compute_q(feature, deep_q_model)
        targets[i][action] = target

    x = np.concatenate([each[0] for each in sample])

    global loss_arr
    loss = deep_q_model.train_on_batch(x, targets)
    loss_arr.append(loss)
    if len(loss_arr) == 100:
        print("loss %s" % str(sum(loss_arr) / len(loss_arr)))
        loss_arr = []


def train_deep_q(training_epoch, epsilon, image_list, bounding_box_list, deep_q_model, vgg16):
    experience = []

    for current_epoch in range(1, training_epoch + 1):

        print("Now starting epoch %d" % current_epoch)
        training_set_size = np.shape(image_list)[0]

        for i in range(training_set_size):
            image = image_list[i]
            ground_truth_box = bounding_box_list[i]
            history = [-1] * history_size
            height, width, channel = np.shape(image)
            current_mask = np.asarray([0, 0, width, height])
            feature = extract_feature(image, history, vgg16)
            end = False
            step = 0
            total_reward = 0

            while not end:
                q_value = compute_q(feature, deep_q_model)
                action = select_action(feature, ground_truth_box, step, q_value, epsilon, current_mask)
                new_mask, reward, end, history = execute_action(action, history, ground_truth_box, current_mask)
                cropped_image = crop_image(image, new_mask)
                new_feature = extract_feature(cropped_image, history, vgg16)
                if len(experience) > max_experience_size:
                    experience = experience[1:]
                    experience.append([feature, action, new_feature, reward, end])
                else:
                    experience.append([feature, action, new_feature, reward, end])

                experience_replay(deep_q_model, experience)
                feature = new_feature
                current_mask = new_mask
                step += 1
                total_reward += reward

            print("Image %d, total reward %i" % (i, total_reward))

        if current_epoch < epsilon_change_steps:
            epsilon -= 0.1
            print("current epsilon is %f" % epsilon)

        keras.models.save_model(deep_q_model, "my_tmp_model.h5")

    return deep_q_model


HUBER_DELTA = 1.0
def smoothL1(y_true, y_pred):
    x = K.abs(y_true - y_pred)
    x = tf.where(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
    return K.sum(x)


def create_q_model():
    model = keras.models.Sequential()
    model.add(Dense(1024, input_shape=(4096 + action_option*history_size,), activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(9, activation='linear'))
    model.compile(loss=smoothL1, optimizer='adam')
    return model


def create_vgg16():
    vgg16 = keras.models.load_model("vgg16.h5")
    vgg16.summary()
    return vgg16


def main():

    # object_number = int(sys.argv[1])
    # training_epoch = sys.argv[2]
    # epsilon = sys.argv[3]
    object_number = 1
    training_epoch = 50
    epsilon = 1
    image_list, bounding_box_list = load_data(object_number)
    deep_q_model = create_q_model()
    vgg16 = create_vgg16()

    # print(np.shape(image_list[0:99]))
    # print(np.shape(bounding_box_list[0:99]))

    trained_model = train_deep_q(training_epoch, epsilon, image_list, bounding_box_list, deep_q_model, vgg16)
    trained_model.save("well_trained_model_100_50.h5")

if __name__ == '__main__':
    main()