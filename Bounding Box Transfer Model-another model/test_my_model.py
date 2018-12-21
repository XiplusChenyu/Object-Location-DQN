from my_q_learning import compute_q, extract_feature, history_size, create_vgg16, compute_mask, crop_image, compute_iou
import os
import cv2
import sys
import numpy as np
import xmltodict
import math
import keras
from keras.layers import Dense, K
import tensorflow as tf
import pandas as pd

history_size = 10
action_option = 9


def get_object_name(object_number):

    try:
        if object_number == 1:
            object_name = "aeroplane"
        elif object_number == 2:
            object_name = "bicycle"
        elif object_number == 3:
            object_name = "bird"
        elif object_number == 4:
            object_name = "boat"
        elif object_number == 5:
            object_name = "bottle"
        elif object_number == 6:
            object_name = "bus"
        elif object_number == 7:
            object_name = "car"
        elif object_number == 8:
            object_name = "chair"
        elif object_number == 9:
            object_name = "cow"
        elif object_number == 10:
            object_name = "diningtable"
        elif object_number == 11:
            object_name = "dog"
        elif object_number == 12:
            object_name = "horse"
        elif object_number == 13:
            object_name = "motorbike"
        elif object_number == 14:
            object_name = "person"
        elif object_number == 15:
            object_name = "pottedplant"
        elif object_number == 16:
            object_name = "sheep"
        elif object_number == 17:
            object_name = "sofa"
        elif object_number == 18:
            object_name = "tvmonitor"
        else:
            raise ValueError
        return object_name

    except ValueError:
        print("Object number invalid")


def read_image_index(object_name, dataset_path):
    index_list = []
    index_file_path = dataset_path + "ImageSets/Main/" + object_name + "_trainval.txt"
    with open(index_file_path, 'r') as f:
        for line in f:
            if "-1" not in line.split(" ")[1]:
                index_list.append(line.split(" ")[0])

    return index_list


def read_image(image_index, dataset_path):

    image_list = []
    image_folder_path = dataset_path + "JPEGImages/"
    for each_image in image_index:
        img = cv2.imread(image_folder_path + each_image + ".jpg")
        image_list.append(img)

    return image_list


def load_annotation(image_index, object_name, dataset_path):
    bounding_box_list = []
    annotattion_path = dataset_path + "Annotations/"
    for each_image in image_index:
        path = annotattion_path + each_image + ".xml"
        xml = xmltodict.parse(open(path, 'rb'))
        xml_objects = xml['annotation']['object']
        if isinstance(xml_objects, list):
            for each_object in xml_objects:

                if each_object["name"] == object_name:
                    xmin = each_object["bndbox"]["xmin"]
                    ymin = each_object["bndbox"]["ymin"]
                    xmax = each_object["bndbox"]["xmax"]
                    ymax = each_object["bndbox"]["ymax"]
                    bounding_box = (int(xmin), int(ymin), int(xmax), int(ymax))
                    bounding_box_list.append(bounding_box)
                    break
        else:
            if xml_objects["name"] == object_name:
                xmin = xml_objects["bndbox"]["xmin"]
                ymin = xml_objects["bndbox"]["ymin"]
                xmax = xml_objects["bndbox"]["xmax"]
                ymax = xml_objects["bndbox"]["ymax"]
                bounding_box = (int(xmin), int(ymin), int(xmax), int(ymax))
                bounding_box_list.append(bounding_box)

    return bounding_box_list


def load_data(object_number):
    dataset_path = "./VOC2012/"
    object_name = get_object_name(object_number)
    image_index = read_image_index(object_name, dataset_path)
    image_list = np.asarray(read_image(image_index, dataset_path))
    bounding_box_list = np.asarray(load_annotation(image_index, object_name, dataset_path))
    np.save(object_name + "_valimage.npy", image_list)
    np.save(object_name + "_valbox.npy", bounding_box_list)

    # print(bounding_box_list[159])
    # cv2.imshow("image", image_list[159])
    # cv2.waitKey(0)
    return image_list, bounding_box_list


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

def test():
    object_number = 1
    image_list, bounding_box_list = load_data(object_number)
    iou = []
    vgg16 = create_vgg16()
    deep_q = create_q_model()
    deep_q.load_weights("my_tmp_model.h5")

    for i in range(0,100):
        bounding_box = bounding_box_list[i]
        image = image_list[i]
        history = [-1] * history_size
        height, width, channel = np.shape(image)
        current_mask = np.asarray([0, 0, width, height])
        feature = extract_feature(image, history, vgg16)
        end = False
        masks = []
        step = 0

        while not end:

            q_value = compute_q(feature, deep_q)

            action = np.argmax(q_value)

            history = history[1:]
            history.append(action)

            if action == 8 or step == 10:
                end = True
                print("end")
                new_mask = current_mask
                cv2.rectangle(image, (int(bounding_box[0]), int(bounding_box[1])),
                              (int(bounding_box[2]), int(bounding_box[3])), (0, 0, 255), 1)
                cv2.imwrite("./result/plane_result%d.jpg" % i, image)
            else:
                new_mask = compute_mask(action, current_mask)

            cropped_image = crop_image(image, new_mask)
            feature = extract_feature(cropped_image, history, vgg16)

            masks.append(new_mask)
            current_mask = new_mask
            cv2.rectangle(image, (int(current_mask[0]), int(current_mask[1])),
                          (int(current_mask[2]), int(current_mask[3])), (0, 255, 0), 1)
            step += 1

        mask = masks[-1]
        iou.append(compute_iou(mask,bounding_box))


    print(sum(iou)/len(iou))
    pd.DataFrame(iou).to_excel('output.xlsx', header=False, index=False)
    # cv2.rectangle(image, (int(mask[0]), int(mask[1])),
    #               (int(mask[2]),int(mask[3])),(0, 255, 0), 2)
    #
    # cv2.imshow("Show", image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

test()