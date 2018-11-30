update: 11/20 add readme
# About SSH and GitHub:
1. How to set **git account** in your computer
```
$ git config --global user.name "<name>"
$ git config --global user.email "<email>"
```
2. How to set up ssh key for your computer
```
$ ssh-keygen -t rsa -C "<your email for this key>"
```
after this you need to input three command:
- where to store the key file: <path>
- set and comfirm password
3. How to get your ssh key
- cd to the location where you store the key
- use vim to open <id_rsa>.pub file
- copy the content, and paste it to where you need to offer you key (GitHub or VM)
<br/>Links:
  https://segmentfault.com/a/1190000002645623
  https://jingyan.baidu.com/article/0320e2c11416cb1b86507b7d.html
# Hierarchical Object Location in Image
Designing a reinforcement learning model with Deep Q-Learning algorithm in order to generate a localization policy which can efficiently locate objects within an image.<br/>
**please pull before you do some works, or please use github branchs**<br/>
**all scripts needed to be update for python3.6**
# Enter VM
**Please** use ssh to copy files between local hosts and VM
# Current structures(11/30):
- root/~#
  - rlpf
    - Hierarchical-Object-Location-in-Image
      - scripts
    - rlen (virtualenv)
    - VOC2012
    - resource (where I put weights and results)
# Run VM background
**Must** use tmux to run: http://louiszhai.github.io/2017/09/30/tmux/
# Requirements (done)
- Python 2.7
- install packages in requirements.txt (already add opencv-python)
- install tensorflow 0.10.0 (with whl file, the file in this repo is for Mac OS X, see https://github.com/tensorflow/tensorflow/blob/v0.10.0rc0/tensorflow/g3doc/get_started/os_setup.md for other versions)
- install tff LiberationMono-Regular
- download VOC2012
- See readme 2 for more
