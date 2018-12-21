#### update: 
> Please add logs here<br/>
12/21 add another model<br/>
12/20 Change the steps num to 8 and start training<br/>
12/14 add an optimizer in RL model<br/>
12/10 update this project with python3.6 and newest tensorflow and Keras<br/>
12/01 update something, hello <br/>
11/30 add git,ssh and tmux <br/>

--------
# About SSH and GitHub (F.Y.I):
#### If you don't want to use github, use ssh and scp operations to copy your files to VM and run what you want.
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
- where to store the key file: "the key path"
- set and comfirm password
3. How to get your ssh key
- cd to the location where you store the key
- use vim to open <id_rsa>.pub file
- copy the content, and paste it to where you need to offer you key (GitHub or VM)
<br/>Links:
  https://segmentfault.com/a/1190000002645623
  https://jingyan.baidu.com/article/0320e2c11416cb1b86507b7d.html
4. After you set-up GitHub, the operation for this projects:
  - cd to git repo in VM, the git repo in our google cloud VM is root#~/rlpj/Hierarchical-Object-Location-in-Image, for example, you want to sync the VM file with our GitHub code:
  ```
  cd ~/rlpj/Hierarchical-Object-Location-in-Image
  git pull origin master
  ```
  Then you can run.
  - or you want to modify in your host:
    - use mkdir to create a folder
    - clone this repo to your folder
    - Now you has local repo
  
```
mkdir rlpj
git clone git@github.com:XiplusChenyu/Hierarchical-Object-Location-in-Image.git
```
5. Sync you modifications and run in VM
- **Before you do your work, update your repo**
  ```
  git pull origin master
  ```
   - After you made changes, for example: you modified a file in ./scripts
  ```
  cd scripts
  git add .       # add you modification in git, '.' means add all file, or you can use <git add sample.py> to add one file
  git commit -m'<your comment for this commit, for example: debug>'
  git push origin master       # You add changes in remote GitHub repo
  ```
  - **Notice**:
  Since we didn't use branchs in github, which means we only has one branch. Please **only push** we you finished debug, and notice others to sync.

-------
# Hierarchical Object Location in Image
Designing a reinforcement learning model with Deep Q-Learning algorithm in order to generate a localization policy which can efficiently locate objects within an image.<br/>
**please pull before you do some works, or please use github branchs**<br/>
**all scripts needed to be update for python3.6**
# Enter VM
**Please** use ssh to copy files between local hosts and VM
# Old structures(11/30):
- root/~#
  - rlpf
    - Hierarchical-Object-Location-in-Image
      - scripts
    - rlen (virtualenv)
    - VOC2012
    - resource (where I put weights and results)
# Run VM background
**Must** use tmux to run: http://louiszhai.github.io/2017/09/30/tmux/
TMUX allow users run several program in terminal without keeping ssh coonections.

# Requirements (done)
- install tff LiberationMono-Regular
- download VOC2012
