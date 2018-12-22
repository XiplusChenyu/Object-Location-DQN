from Read_Image import *
from Read_Annotations import *
from Setting import *
from IoU import *
from Vgg16 import *
from RL import *
from View import *
from Agent import *
import argparse
import PIL
import time
from PIL import Image, ImageDraw


def averagenum(num):
    nsum = 0
    for i in range(len(num)):
        nsum += num[i]
    return nsum / len(num)


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
    test_count = test_size
    IoU_storage = []

    for image_index in range(len(image_names)):
        image_index += 10
        if not labels[image_index] == '1':
            continue
        if not test_count > 0:
            break
        test_count -= 1
        image_name = image_names[image_index]
        image = np.array(obtain_image(image_name))
        image_for_search = image
        tmp = 0

        '''Create Masks'''
        annotation = annotations[image_index]
        gt_masks = create_masks(annotation, image.shape)
        object_num = np.size(annotation[:, 0])
        size_mask = (image.shape[0], image.shape[1])
        original_shape = size_mask
        region_mask = np.ones([image.shape[0], image.shape[1]])

        '''The draw image part for each step'''
        background = Image.new('RGBA', (10000, 2000), (255, 255, 255, 255))
        draw = ImageDraw.Draw(background)

        '''The draw image part for bounding box move'''
        background2 = Image.fromarray(image)
        draw2 = ImageDraw.Draw(background2)
        color_number = 0

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
        q_value = [[0]]

        region_image = image_for_search
        region_mask = np.ones([image.shape[0], image.shape[1]])

        while (step < step_num_test) and absolute_status:
            # box_color = (color_number, 0, 100)
            box_color = (255, 0, 0)
            # color_number = 0
            iou = 0

            '''Set History Vector'''
            # we init history vector as we are going to find another object
            history_vector = np.zeros([24])
            status = 1

            draw_sequences_test(step, action, q_value, draw, region_image, background, path_testing_folder,
                                region_mask, image_name, bool_draw_test)

            size_mask = (image.shape[0], image.shape[1])
            original_shape = size_mask
            region_mask = np.ones(size_mask)

            state = get_state(region_image, history_vector, model_vgg)

            while (status == 1) and (step < step_num_test):
                step += 1
                q_value = rl_model.predict(state.T, batch_size=1)
                # print (q_value)
                action = (np.argmax(q_value)) + 1
                box_area = ((int(offset[1] + size_mask[1]), int(offset[0] + size_mask[0])),
                            (int(offset[1]), int(offset[0])))

                if action == 6:
                    box_color = (0, 0, 255)
                    draw_sequences_test_box(draw2, background2, box_area, box_color, path_testing_folder, image_name,
                                            bool_draw_test)
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
                    box_area = ((int(offset[1] + size_mask[1]), int(offset[0] + size_mask[0])),
                                (int(offset[1]), int(offset[0])))

                    draw_sequences_test(step, action, q_value, draw, region_image, background, path_testing_folder,
                                        region_mask, image_name, bool_draw_test)

                    draw_sequences_test_box(draw2, background2, box_area, box_color, path_testing_folder, image_name,
                                            bool_draw_test)
                    # color_number += 30
                    # box_color = (color_number, 0, 100)

                history_vector = update_history_vector(history_vector, action)
                new_state = get_state(region_image, history_vector, model_vgg)
                state = new_state

        tmp = []
        for index_o in range(object_num):
            gt_mask = gt_masks[:, :, index_o]
            happy = iou_calculator(region_mask, gt_mask)
            tmp.append(happy)

        iou_result = max(tmp)
        IoU_storage.append(iou_result)

    print(averagenum(IoU_storage))


test_size = int(input('Enter the Test Size (How Many Pics):\n'))
bool_draw_test = int(input('Whether to store the visualization pics (1 for Yes/ 0 for No): \n'))

main()














