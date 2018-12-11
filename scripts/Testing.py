from Read_Image import *
from Read_Annotations import *
from Setting import *
from Metrics import *
from Vgg16 import *
from RL import *
from View import *
from Agent import *
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
    model_vgg, rl_model = build_model()

    for image_index in range(len(image_names)):
        if not labels[image_index] == '1':
            continue

        image_name = image_names[image_index]
        image = np.array(obtain_image(image_name))
        image_for_search = image

        '''The draw image part'''
        background = Image.new('RGBA', (10000, 2000), (255, 255, 255, 255))
        draw = ImageDraw.Draw(background)

        '''Create Masks'''
        annotation = annotations[image_index]
        gt_masks = create_masks(annotation, image.shape)
        size_mask = (image.shape[0], image.shape[1])
        original_shape = size_mask
        region_mask = np.ones([image.shape[0], image.shape[1]])

        '''object within the image'''
        objects = annotation[:, 0]

        '''Crop Initialization'''
        offset = (0, 0)

        '''RL Initialization'''
        # absolute status is a boolean we indicate if the agent will continue
        # searching object or not. If the first object already covers the whole
        # image, we can put it at 0 so we do not further search there
        absolute_status = True
        action = 0
        step = 0
        q_value = 0

        region_image = image_for_search
        region_mask = np.ones([image.shape[0], image.shape[1]])

        while (step < step_num_test) and absolute_status:
            iou = 0

            '''Set History Vector'''
            # we init history vector as we are going to find another object
            history_vector = np.zeros([24])
            state = get_state(region_image, history_vector, model_vgg)
            status = 1

            draw_sequences_test(step, action, q_value, draw, region_image, background, path_testing_folder,
                                region_mask, image_name, bool_draw_test)

            size_mask = (image.shape[0], image.shape[1])
            original_shape = size_mask
            region_mask = np.ones(size_mask)

            while (status == 1) and (step < step_num_test):
                step += 1
                q_value = rl_model.predict(state.T, batch_size=1)
                action = (np.argmax(q_value)) + 1

                if action == 6:
                    offset = (0, 0)
                    status = 0

                    if step == 1:
                        absolute_status = False
                    if only_first_object == 1:
                        absolute_status = False

                    image_for_search = mask_image_with_mean_background(region_mask, image_for_search)
                    region_image = image_for_search

                else:
                    region_image, region_mask, offset, size_mask = agent_move_mask(action, original_shape,
                                                                                   size_mask, offset, region_image)

                    print(region_image.shape)
                    draw_sequences_test(step, action, q_value, draw, region_image, background, path_testing_folder,
                                        region_mask, image_name, bool_draw_test)

                history_vector = update_history_vector(history_vector, action)
                new_state = get_state(region_image, history_vector, model_vgg)
                state = new_state


test_size = int(input('Enter the Test Size (How Many Pics):\n'))
bool_draw_test = int(input('Whether to store the visualization pics (1 for Yes/ 0 for No): \n'))

main()














