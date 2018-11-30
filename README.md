# Hierarchical Object Location in Image
Designing a reinforcement learning model with Deep Q-Learning algorithm in order to generate a localization policy which can efficiently locate objects within an image.
# Enter VM
**Please** use ssh to copy files between local hosts and VM
# Current structures:
- root/~#
  - rlpf
    - Hierarchical-Object-Location-in-Image
      - scripts
    - rlen (virtualenv)
    - VOC2012
    - resource (where I put weights and results)
# Run VM background
**Must** use tmux to run: http://louiszhai.github.io/2017/09/30/tmux/
# Requirements
- Python 2.7
- install packages in requirements.txt (already add opencv-python)
- install tensorflow 0.10.0 (with whl file, the file in this repo is for Mac OS X, see https://github.com/tensorflow/tensorflow/blob/v0.10.0rc0/tensorflow/g3doc/get_started/os_setup.md for other versions)
- install tff LiberationMono-Regular
- download VOC2012
- See readme 2 for more
