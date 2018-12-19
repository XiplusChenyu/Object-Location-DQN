from keras.preprocessing import image
from Setting import *
from PIL import Image
import numpy as np

results = np.zeros([5, 1])
last_matrix = np.zeros([5])
replay = [[] for i in range(20)]
print(replay)