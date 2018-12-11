Pull Before You Push!!!<br/>
Current issues:
- There is no model weights for Vgg16
- The training results is weired
# Hierarchical-Object-Location-in-Image
A pathetic project<br/>
**See reference.md for reference**<br/>
Image Zooming Method<br/>
## Current Setting:
Training Object: bird
## Requirement
All up to date</br>
Python 3.6
## The Current Structure:
- Project
  - VOC2012
  - img_test (save test pics here)
  - img_train (save training process pics here)
  - vgg16_weight.h5 (pre-trained vgg16 model) ***need to be added***
  - models_image_zooms (save Q-network models here)
    - tmp models for epochs
    - final model
  - Program (This repo)
    - readme and other guides
    - scripts
      - Training.py (use this for training)
      - Testing.py (use this for testing)
      - Setting.py (use this for setting paras)
      - Others
