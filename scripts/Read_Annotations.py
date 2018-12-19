import xml.etree.ElementTree as et
import numpy as np
from Setting import *


# @ assign an id for each class
def get_id(class_name):
    if class_name == 'aeroplane':
        return 1
    elif class_name == 'bicycle':
        return 2
    elif class_name == 'bird':
        return 3
    elif class_name == 'boat':
        return 4
    elif class_name == 'bottle':
        return 5
    elif class_name == 'bus':
        return 6
    elif class_name == 'car':
        return 7
    elif class_name == 'cat':
        return 8
    elif class_name == 'chair':
        return 9
    elif class_name == 'cow':
        return 10
    elif class_name == 'diningtable':
        return 11
    elif class_name == 'dog':
        return 12
    elif class_name == 'horse':
        return 13
    elif class_name == 'motorbike':
        return 14
    elif class_name == 'person':
        return 15
    elif class_name == 'pottedplant':
        return 16
    elif class_name == 'sheep':
        return 17
    elif class_name == 'sofa':
        return 18
    elif class_name == 'train':
        return 19
    elif class_name == 'tvmonitor':
        return 20


def obtain_ground_truth(name, path=path_dataset):
    file = path + '/Annotations/' + name + '.xml'
    with open(file, 'r') as f:
        xml_contains = f.read()
        root = et.XML(xml_contains)
        names = []
        x_min = []
        x_max = []
        y_min = []
        y_max = []
        for child in root:
            if child.tag == 'object':
                for child2 in child:
                    if child2.tag == 'name':
                        names.append(child2.text)
                    elif child2.tag == 'bndbox':
                        for child3 in child2:
                            if child3.tag == 'xmin':
                                x_min.append(child3.text)
                            elif child3.tag == 'xmax':
                                x_max.append(child3.text)
                            elif child3.tag == 'ymin':
                                y_min.append(child3.text)
                            elif child3.tag == 'ymax':
                                y_max.append(child3.text)
        results = np.zeros([np.size(names), 5])
        for i in range(np.size(names)):
            results[i][0] = get_id(names[i])
            results[i][1] = x_min[i]
            results[i][2] = x_max[i]
            results[i][3] = y_min[i]
            results[i][4] = y_max[i]
        return results.astype("uint32")


def gain_annotations(names, path=path_dataset):
    annotations = []
    for name in names:
        annotations.append(obtain_ground_truth(name, path))
    return annotations


# @Create masks for all pictures within one picture
def create_masks(annotation, image_shape):
    object_num = len(annotation)
    masks = np.zeros([image_shape[0], image_shape[1], object_num])
    for i in range(object_num):

        masks[annotation[i, 3]:annotation[i, 4],
              annotation[i, 1]:annotation[i, 2], i] = 1

    return masks



