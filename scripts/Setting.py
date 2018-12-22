import numpy as np

# @ Epoch set:
# epochs_id = 0
# epochs = 10
# train_size = 100
# test_size = 10


# @ 1 if you want to obtain visualizations of the search for objects
# bool_draw = 0
# bool_draw_test = 1

# @ Which class to train:
class_id = 3

# @which class to test
test_dataset_name = 'bird_val'
test_class_id = 3
test_model_name = '/model'+str(class_id)+'.h5'
only_first_object = 1  # @whether only search the first object while testing

# @ The dir info:
path_dataset = './../../VOC2012'
path_vgg16 = './../../vgg16_weights.h5'
path_testing_folder = './../../img_test'
path_training_folder = './../../img_train'
path_model = "./../../models_image_zooms"
path_test_dataset = './../../VOC2012'
use_vgg_weight = False

# @ RL num
step_num = 10
step_num_test = 8
gamma = 0.90
# Each replay memory (one for each possible category) has a capacity of 100 experiences
buffer_experience_replay = 500
# Pointer to where to store the last experience in the experience replay buffer,
# actually there is a pointer for each PASCAL category, in case all categories
# are trained at the same time
# h = np.zeros([20])
batch_size = 100


# @ Model Part
epsilon = 0.9
# Visual descriptor size
visual_descriptor_size = 25088
# Different actions that the agent can do
number_of_actions = 6
# Actions captures in the history vector
actions_of_history = 4


# Scale of sub-region for the hierarchical regions (to deal with 2/4, 3/4)
scale_subregion = float(3) / 4
scale_mask = float(1) / (scale_subregion * 4)
# IoU required to consider a positive detection
iou_threshold = 0.5
