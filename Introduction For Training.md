# Introduction
Brief introduction for Training.py<br/>

<a href='https://www.cnblogs.com/pinard/p/9714655.html'>**Useful source**</a> for understanding Deep Q-Learning
Algorithm in this program.

## Neural Networks
- Vgg16
```
<predict>
input:
image, history actions
output:
current state (image feature + history vectors)
```
- Q-network:
```
<predict>
input: 
current state (output of Vgg16)
output:
the q value vector for current state
```
## Work Flow
Ignore the data processing, model saving and the visualization part.<br/>

### The input arguments:
- Training Size: # pics to train with
- Checkpoint(disabled): whether save models as one epoch finished
- Epochs_id(disabled): which epoch to resume training, if you saved previous models
- Epochs: # epochs to train
- Bool_draw(disabled): whether save visualization pics when training 

### Several Factors:
- **Mask**: We use a 1/0 mask to simulate the cropping.
- **State**: We defined current cropped region as current state
- **Agent**:
  - Actions:
    ```python
     if action == 0:
        return "START"
    if action == 1:
        return 'up-left'
    elif action == 2:
        return 'up-right'
    elif action == 3:
        return 'down-left'
    elif action == 4:
        return 'down-right'
    elif action == 5:
        return 'center'
    elif action == 6:
        return 'TRIGGER'
    ```
  - Reward: We defined the reward based on the IoU of our **cropped region** and the object we want to find:  
    ```python
    # for movement
    def get_reward_movement(iou, new_iou):
    if new_iou > iou:
        reward = 1
    else:
        reward = - 1
    return reward
    # for terminal action
    def get_reward_trigger(new_iou):
    if new_iou > iou_threshold:
        reward = 3
    else:
        reward = - 3
    return reward
    ```
- The basic flow:  
  ```
    Start one epoch
      ----> Choose one Picture:
         ----> Apply RL Algorithm on each object within one picture
           ----> We execute certain steps / or end with located object
    ```
    
### Tricks
#### replay
As is known to all, DQN need experience replay, we create a list <replay> as the experience pool. 

#### last_matrix
 **Notice** that though we intended to **search for one object** within the picture for each RL algorithm execution, we may **actually change** the object during steps.

**For example:** we want to locate bird1 in an image, but at a step we may cropped a region which actually locate bird2. Thus we use a 1d matrix to store IoUs (current region over all objects)
```python
last_matrix = np.zeros(len(objects))
```
#### objects_available
There might be objects which already fill up most of the image, we won't train them to gain a faster convergence.
```
for object_index2 in range(len(objects)):
    overlap = overlap_calculator(old_region_mask, gt_masks[:, :, object_index2])
    if overlap > 0.60:
        objects_available[object_index2] = 0
```
#### different greedy policy:
```
if random.random() < epsilon_value:
                    action = np.random.randint(1, 7)

else:
    action = (np.argmax(q_value)) + 1
```
It's not a epsilon greedy, and we decrease <epsilon_value> each epoch iteration. We start with highly exploration and end up with greedy.  

### The Work Flow
```
Initialization:
create history vector, stores past 4 steps' actions
create state from model_vgg
create objects_available_list

set: 
  status = 1
  steps = 0
  action = 0 ("Start" Action)
  
Steps:
<break the episode when: status = 0, 
                         steps overflow, 
                         no available objects remaining.>
<Agent Moving>
1. Get the state by calling get_state function
2. Call rl_model.predict on current state, the Q network generating Q value for each action (except the "start" action)
3. Using Greedy Policy to choose an action, calculate reward.
4. Update state:
   Set current cropped subregion as new state
   
<Update the Q network>
5. Put <state, Action, new_state> in replay list
6. Sample n instances from replay list (experience pool) as a mini-batch, we caculate the Q-value for current target.
7. We fit the RL model based on step 6.
```
```python
the update functions for update Q-network:

# Retrieve past experience:
mini_batch = random.sample(replay[category], batch_size)

x_train = [] # Record the state
y_train = [] # Record the target Q-value

for memory in mini_batch:
    old_state, action, reward, new_state = memory

    old_q_value = rl_model.predict(old_state.T, batch_size=1)
    new_q_value = rl_model.predict(new_state.T, batch_size=1)
    maxQ = np.max(new_q_value)

    y = np.zeros([1, 6])
    y = old_q_value.T

    if not action == 6: # If it's the end step
        update = (reward + (gamma * maxQ))
    else:
        update = reward

    y[action - 1] = update
    x_train.append(old_state)
    y_train.append(y)

# Update the Q-network weights
x_train = np.array(x_train).astype("float32")
y_train = np.array(y_train).astype("float32")
x_train = x_train[:, :, 0]
y_train = y_train[:, :, 0]

rl_model.fit(x_train, y_train, batch_size=batch_size, epochs=1, verbose=0)
```


    





                            





