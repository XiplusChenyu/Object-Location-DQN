from Read_Image import *
from Read_Annotations import *
from Setting import *
from IoU import *
from Vgg16 import *
from RL import *
from View import *
import argparse
import PIL
from Agent import *
import time
from PIL import Image, ImageDraw

'''
Part of reading images
info: 
read images, gain image ground truth of bounding box, gain image id
'''


def build_model():
    #  build vgg16 model here
    model_vgg = vgg16_model()

    #  build reinforcement model here!
    rl_model = q_network()
    return model_vgg, rl_model
    # if epochs_id_input == 0:
    #     rl_models = qn_pascal("0", class_id)
    # else:
    #     rl_models = qn_pascal(path_model, class_id)
    # return model_vgg, rl_models


def main():
    # model_vgg, rl_models = build_model(epochs_id)
    model_vgg, rl_model = build_model()
    epsilon_value = epsilon

    # Init replay memories
    replay = []
    h = 0
    # RL reset
    # reward = 0

    # replay = [[] for x in range(20)]

    for epoch_index in range(epochs_id, epochs + epochs_id):
        print('start epoch: ' + str(epoch_index))
        print(epsilon_value)
        time1 = time.time()

        # @Use Train_size to fasten the learning process
        for pic_index in range(train_size):
            masked = 0  # Record if we have masked this picture, i.e, if we have trained one object within this picture
            not_finished = 1  # Record if we already done with this picture

            # @ Read picture info
            image_name = image_names[pic_index]
            picture = np.array(obtain_image(image_name))
            picture_annotation = annotations[pic_index]

            # @Create mask area and obtain objects in pictures
            gt_masks = create_masks(picture_annotation, picture.shape)  # Ground truth For several objects
            picture_size = (picture.shape[0], picture.shape[1])

            # print(picture_annotation, len(picture_annotation), picture_array.shape, objects)
            objects = picture_annotation[:, 0]
            objects_available = np.ones(len(objects))

            # @For each objects in this picture:
            for object_index in range(len(objects)):

                # @skip Objects not belongs to Class
                if not objects[object_index] == class_id:
                    continue

                '''visualization here'''
                background = Image.new('RGBA', (10000, 2500), (255, 255, 255, 255))
                draw = ImageDraw.Draw(background)

                '''RL paras init'''
                region_image = picture
                step = 0
                # new_iou = 0
                offset = (0, 0)

                '''Start to initialize the RL part'''
                last_matrix = np.zeros(len(objects))
                history_vector = np.zeros([24])  # @ the history vectors is the 6 actions*pass 4 steps

                '''Set Ground truth mask for one object'''
                mask_size = picture_size
                original_shape = mask_size
                region_mask = np.ones(picture_size)
                old_region_mask = region_mask

                ''' Test if the pictures is masked before'''
                if masked == 1:
                    for object_index2 in range(len(objects)):
                        overlap = overlap_calculator(old_region_mask, gt_masks[:, :, object_index2])
                        if overlap > 0.60:
                            objects_available[object_index2] = 0

                if np.count_nonzero(objects_available) == 0:
                    not_finished = 0

                '''The initial status'''
                # status indicates whether the agent is still alive and has not triggered the terminal action
                status = 1
                action = 0
                reward = 0

                # @Calculate IoU for first state here
                iou, new_iou, last_matrix, iou_index = iou_iteration(gt_masks, region_mask,
                                                                     objects, class_id, last_matrix,
                                                                     objects_available)

                gt_mask = gt_masks[:, :, iou_index]
                # the first object which overlaps most with the whole picture

                iou = new_iou

                # @ Use vgg16 to get the state descriptor
                state = get_state(region_image, history_vector, model_vgg)

                while (status == 1) & (step < step_num) & not_finished:
                    # @Calculate Q value for each steps
                    q_value = rl_model.predict(state.T, batch_size=1)
                    # @q_value is 6-dims for each action

                    background = draw_sequences(epoch_index, object_index, step, action, draw, region_image,
                                                background, path_training_folder, iou, reward, gt_mask, region_mask,
                                                image_name, bool_draw)
                    step += 1

                    # we force terminal action in case actual IoU is higher than 0.5, to train faster the agent
                    if new_iou > 0.5 and epoch_index < 100:
                        action = 6
                    # epsilon-greedy policy
                    # else:
                    #     action = epsilon_greedy(q_value, epsilon_value)
                    #     print('greedy')
                    #     print(q_value)

                    # epsilon-greedy policy
                    elif random.random() < epsilon_value:
                        action = np.random.randint(1, 7)

                    else:
                        action = (np.argmax(q_value)) + 1

                    # @ The terminal state
                    if action == 6:
                        iou, new_iou, last_matrix, iou_index = iou_iteration(gt_masks, region_mask,
                                                                             objects, class_id, last_matrix,
                                                                             objects_available)
                        gt_mask = gt_masks[:, :, iou_index]
                        reward = get_reward_trigger(new_iou)

                        background = draw_sequences(epoch_index, object_index, step, action, draw, region_image,
                                                    background, path_training_folder, iou, reward, gt_mask, region_mask,
                                                    image_name, bool_draw)
                        step += 1

                    # @The movement actions
                    else:
                        region_image, region_mask, offset, mask_size = agent_move_mask(action, original_shape,
                                                                                       mask_size, offset, region_image)
                        # Replaced by function -- See agent.py
                        # region_mask = np.zeros(original_shape)
                        # mask_size = (mask_size[0] * scale_subregion, mask_size[1] * scale_subregion)
                        #
                        # offset_aux = (0, 0)  # @The action == 1
                        # if action == 2:
                        #     offset_aux = (0, mask_size[1] * scale_mask)
                        #     offset = (offset[0], offset[1] + mask_size[1] * scale_mask)
                        # elif action == 3:
                        #     offset_aux = (mask_size[0] * scale_mask, 0)
                        #     offset = (offset[0] + mask_size[0] * scale_mask, offset[1])
                        # elif action == 4:
                        #     offset_aux = (mask_size[0] * scale_mask,
                        #                   mask_size[1] * scale_mask)
                        #     offset = (offset[0] + mask_size[0] * scale_mask,
                        #               offset[1] + mask_size[1] * scale_mask)
                        # elif action == 5:
                        #     offset_aux = (mask_size[0] * scale_mask / 2,
                        #                   mask_size[0] * scale_mask / 2)
                        #     offset = (offset[0] + mask_size[0] * scale_mask / 2,
                        #               offset[1] + mask_size[0] * scale_mask / 2)
                        #
                        # # @Update the status here, crop  the image, and update the region mask:
                        # region_image = region_image[int(offset_aux[0]):int(offset_aux[0] + mask_size[0]),
                        #                             int(offset_aux[1]): int(offset_aux[1] + mask_size[1])]
                        #
                        # region_mask[int(offset[0]): int(offset[0] + mask_size[0]),
                        #             int(offset[1]): int(offset[1] + mask_size[1])] = 1

                        iou, new_iou, last_matrix, iou_index = iou_iteration(gt_masks, region_mask,
                                                                             objects, class_id, last_matrix,
                                                                             objects_available)
                        gt_mask = gt_masks[:, :, iou_index]

                        background = draw_sequences(epoch_index, object_index, step, action,
                                                    draw, region_image, background,
                                                    path_training_folder, iou, reward, gt_mask, region_mask, image_name,
                                                    bool_draw)

                        reward = get_reward_movement(iou, new_iou)
                        iou = new_iou

                    history_vector = update_history_vector(history_vector, action)
                    new_state = get_state(region_image, history_vector, model_vgg)

                    # Experience replace storage, stores past experience as target
                    if len(replay) < buffer_experience_replay:
                        replay.append((state, action, reward, new_state))
                    else:
                        if h < (buffer_experience_replay - 1):
                            h += 1
                        else:
                            h = 0

                        # h_aux = int(h[category])
                        replay[h] = (state, action, reward, new_state)
                        mini_batch = random.sample(replay, batch_size)  # arbitrary choose mini_batch

                        x_train = []
                        y_train = []

                        for memory in mini_batch:
                            old_state, action, reward, new_state = memory

                            old_q_value = rl_model.predict(old_state.T, batch_size=1)
                            new_q_value = rl_model.predict(new_state.T, batch_size=1)

                            max_q_value = np.max(new_q_value)
                            y = old_q_value.T

                            if not action == 6:
                                update = (reward + (gamma * max_q_value))
                            else:
                                update = reward

                            y[action - 1] = update
                            x_train.append(old_state)
                            y_train.append(y)

                        x_train = np.array(x_train).astype("float32")
                        y_train = np.array(y_train).astype("float32")
                        x_train = x_train[:, :, 0]
                        y_train = y_train[:, :, 0]

                        rl_model.fit(x_train, y_train, batch_size=batch_size, epochs=1, verbose=0)

                        # rl_models[0][category] = rl_model

                        state = new_state

                    if action == 6:
                        status = 0
                        masked = 1
                        picture = mask_image_with_mean_background(gt_mask, picture)
                    else:
                        masked = 0
                objects_available[object_index] = 0

        # if epsilon_value > 0.1:
        #     epsilon_value -= 0.1

        model_name = path_model + '/model' + str(class_id) + '.h5'
        model_name2 = path_model + '/model' + str(class_id) + str(epoch_index) + '.h5'
        rl_model.save_weights(model_name, overwrite=True)
        if epoch_index % 3 == 0:
            rl_model.save_weights(model_name2, overwrite=True)


        # for t in range(np.size(rl_models)):
        #     try:
        #         if t == (class_id - 1):
        #             if checkpoint == 1:
        #                 string = path_model + '/model' + str(class_id) + '_epoch_' + str(epoch_index) + '.h5'
        #                 string2 = path_model + '/model' + str(class_id) + '.h5'  # Record the newest one
        #                 model = rl_models[0][t]
        #                 model.save_weights(string, overwrite=True)
        #                 model.save_weights(string2, overwrite=True)
        #             else:
        #                 # string = path_model + '/model' + str(class_id) + '_epoch_' + str(epoch_index) + '.h5'
        #                 string2 = path_model + '/model' + str(class_id) + '.h5'  # Record the newest one
        #                 model = rl_models[0][t]
        #                 # model.save_weights(string, overwrite=True)
        #                 model.save_weights(string2, overwrite=True)
        #
        #     except:
        #         print('model ' + str(class_id) + ' epoch ' + str(epoch_index) + "failed")
        #         pass

        print('end epoch: ' + str(epoch_index))
        time2 = time.time()
        duration = (time2 - time1) / 60
        print('time cost = ' + str(round(duration, 3)) + "min")


train_size = int(input('Enter the Training Size (How Many Pics):\n').strip())

# checkpoint = 0
# epochs_id = 0
# checkpoint = int(input('Do you want to save check points for each epoch? 1 for yes, 0 for no:\n').strip())
# epochs_id = int(input("Enter the epochs_id to resume:\n").strip())

epochs = int(input("Enter the epochs num:\n").strip())
bool_draw = int(input('Whether to store the visualization pics (1 for Yes/ 0 for No): \n').strip())
# class_id = int(input('Choose Object id (default = 3, bird): \n'))

dataset = 'trainval'
image_names = load_image_names(dataset)
if train_size > len(image_names):
    train_size = len(image_names)
    print('training size overflow')

# @The structure of annotations is [object1[id, x_min, x_max, y_min, y_max], object2...]
annotations = gain_annotations(image_names)
print('images reading done')
print('training size = ' + str(train_size))

epochs_id = 9
epsilon = 0.1
main()