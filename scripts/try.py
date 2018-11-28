from features import *
from image_helper import *
from parse_xml_annotations import *
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt


path_voc = "./VOC2012"
two_databases = 0
image_names = np.array([load_images_names_in_data_set('trainval', path_voc)])
images = get_all_images(image_names, path_voc)

for j in range(np.size(image_names)):
    masked = 0
    not_finished = 1
    image = np.array(images[j])
    image_name = image_names[0][j]
    annotation = get_bb_of_gt_from_pascal_xml_annotation(image_name, path_voc)

    gt_masks = generate_bounding_box_from_annotation(annotation, image.shape)
    array_classes_gt_objects = get_ids_objects_from_annotation(annotation)
    region_mask = np.ones([image.shape[0], image.shape[1]])
    shape_gt_masks = np.shape(gt_masks)
    available_objects = np.ones(np.size(array_classes_gt_objects))

