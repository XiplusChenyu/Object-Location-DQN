# Hierarchical Object Detection with Deep Reinforcement Learning (Part)
Original Source: https://github.com/XiplusChenyu/detection-2016-nipsws

## Code Instructions

This python code enables to both train and test each of the two models proposed in the paper. The image zooms model extracts features for each region visited, whereas the pool45 crops model extracts features just once and then ROI-pools features for each subregion. In this section we are going to describe how to use the code. The code uses Keras framework library. If you are using a virtual environment, you can use the requirements.txt provided.


First it is important to notice that this code is already an extension of the code used for the paper. During the training stage, we are not only considering one object per image, we are also training for other objects by covering the already found objects with the mean of VGG-16, inspired by what Caicedo et al. did on Active Object Localization with Deep Reinforcement Learning.

### Setup

First of all the weights of VGG-16 should be downloaded from the following link [VGG-16 weights]. If you want to use some pre-trained models for the Deep Q-network, they can be downloaded in the following link [Image Zooms model]. Notice that these models could lead to different results compared to the ones provided in the paper, due that these models are already trained to find more than one instance of planes in the image. You should also create two folders in the root of the project, called models_image_zooms and models_pool45_crops, and store inside them the corresponding weights.


[VGG-16 weights]: http://imatge.upc.edu/web/sites/default/files/projects/deeplearning/public/detection-2016-nipsws/vgg16_weights.h5
[Image Zooms model]: http://imatge.upc.edu/web/sites/default/files/projects/deeplearning/public/detection-2016-nipsws/model_image_zooms_2


### Usage

##### Training

We will follow as example how to train the Image Zooms model, that is the one that achieves better results. The instructions are equal for training the Pool45 Crops model. The script is image_zooms_training.py, and first the path to the database should be configured. The default paths are the following:

    # path of PASCAL VOC 2012 or other database to use for training
    path_voc = "./VOC2012/"
    # path of other PASCAL VOC dataset, if you want to train with 2007 and 2012 train datasets
    path_voc2 = "./VOC2007/"
    # path of where to store the models
    path_model = "../models_image_zooms"
    # path of where to store visualizations of search sequences
    path_testing_folder = '../testing_visualizations'
    # path of VGG16 weights
    path_vgg = "../vgg16_weights.h5"

But you can change them to point to your own locations.

The training of the models enables checkpointing, so you should indicate which epoch you are going to train when running the script. If you are training it from scratch, then the training command should be:

python image_zooms_training.py -n 0

There are many options that can be changed to test different configurations:

**class_object**: for which class you want to train the models. We have trained it for planes, and all the experiments of the paper are run on this class, but you can test other categories of pascal, also changing appropiately the training databases.

**number_of_steps**: For how many steps you want your agent to search for an object in an image.

**scale_subregion**: The scale of the subregions in the hierarchy, compared to its ancestor. Default value is 3/4, that denoted good results in our experiments, but it can easily be set. Take into consideration that the subregion scale and the number of steps is very correlated, if the subregion scale is high, then you will probably require more steps to find objects.

**bool_draw**: This is a boolean, that if it is set to 1, it stores visualizations of the sequences for image searches.

At each epoch the models will be saved in the models_image_zooms folder.

##### Testing

To test the models, you should use the script image_zooms_testing.py. You should also configure the paths to indicate which weights you want to use, in the same manner as in the training stage. In this case, you should only run the command python image_zooms_testing.py. It is recommended that for testing you put bool_draw = 1, so you can observe the visualizations of the object search sequences. There is the option to just search for a single object in each image, to reproduce the same results of our paper, by just setting the boolean only_first_object to 1.
