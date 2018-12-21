# Hierarchical Object Detection with Deep Reinforcement Learning (Part)
Original Source: https://imatge-upc.github.io/detection-2016-nipsws/

## Code Instructions

First it is important to notice that this code is already an extension of the code used for the paper. During the training stage, we are not only considering one object per image, we are also training for other objects by covering the already found objects with the mean of VGG-16, inspired by what Caicedo et al. did on Active Object Localization with Deep Reinforcement Learning.

### Setup (For Old Version)

First of all the weights of VGG-16 should be downloaded from the following link [VGG-16 weights]. If you want to use some pre-trained models for the Deep Q-network, they can be downloaded in the following link [Image Zooms model]. Notice that these models could lead to different results compared to the ones provided in the paper, due that these models are already trained to find more than one instance of planes in the image. You should also create two folders in the root of the project, called models_image_zooms and models_pool45_crops, and store inside them the corresponding weights.


[VGG-16 weights]: http://imatge.upc.edu/web/sites/default/files/projects/deeplearning/public/detection-2016-nipsws/vgg16_weights.h5
[Image Zooms model]: http://imatge.upc.edu/web/sites/default/files/projects/deeplearning/public/detection-2016-nipsws/model_image_zooms_2


### Usage

##### Training

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

# Bounding Box Transfer Model (Part)
Original Source: https://github.com/tashrifbillah/Object-Detection
