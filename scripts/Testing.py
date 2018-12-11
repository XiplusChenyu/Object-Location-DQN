from Read_Image import *
from Read_Annotations import *
from Setting import *
from Metrics import *
from Vgg16 import *
from RL import *
from View import *
import argparse
import PIL
import time
from PIL import Image, ImageDraw


def build_model(weights_path=path_model, model_name=test_model_name):
    #  build vgg16 model here
    model_vgg = vgg16_model()
    #  build reinforcement model here:
    rl_model = q_network(weights_path + model_name)
    return model_vgg, rl_model


def load_info(dataset_name=test_dataset_name):
    image_names = load_image_names(dataset_name)
    annotations = gain_annotations(image_names)
    labels = load_image_labels(dataset_name)  # whether the object is in this pic, 1 for yes, -1 for no
    return image_names, labels, annotations


def main():
    image_names, labels, annotations = load_info()
    for image_index in range(len(image_names)):
        if not labels[image_index] == '1':
            continue
        image = np.array(obtain_image(image_names[image_index]))

        '''The draw image part'''
        background = Image.new('RGBA', (10000, 2000), (255, 255, 255, 255))
        draw = ImageDraw.Draw(background)
        annotation = annotations[image_index]
        gt_masks = create_masks(annotation, image.shape)

        '''object within the image'''
        objects = annotation[:, 0]









