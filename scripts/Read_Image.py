from keras.preprocessing import image
from Setting import *


# @ load image names in dataset
def load_image_names(dataset, path=path_dataset):
    file_path = path + '/ImageSets/Main/' + dataset + '.txt'
    with open(file_path) as f:
        lines = f.readlines()
        image_names = [line.split(None, 1)[0].strip() for line in lines]
    return image_names


def obtain_images(names, path=path_dataset):
    images = []
    for name in names:
        image_path = path + '/JPEGImages/' + name + '.jpg'
        p = image.load_img(image_path, False)
        images.append(p)
    return images


def obtain_image(name, path=path_dataset):
    image_path = path + '/JPEGImages/' + name + '.jpg'
    p = image.load_img(image_path, False)
    return p


def load_image_labels(dataset, path=path_dataset):
    file_path = path + '/ImageSets/Main/' + dataset + '.txt'
    with open(file_path) as f:
        lines = f.readlines()
        image_labels = [line.split(None, 1)[1].strip() for line in lines]
    return image_labels


def mask_image_with_mean_background(mask_object_found, img):
    new_image = img
    size_image = np.shape(mask_object_found)
    for j in range(size_image[0]):
        for i in range(size_image[1]):
            if mask_object_found[j][i] == 1:
                    new_image[j, i, 0] = 103.939
                    new_image[j, i, 1] = 116.779
                    new_image[j, i, 2] = 123.68
    return new_image
