import numpy as np
import cv2

'''This part calculate several paras used in RL'''


def iou_calculator(img_mask, gt_mask):
    gt_mask *= 1.0
    img_and = cv2.bitwise_and(img_mask, gt_mask)
    img_or = cv2.bitwise_or(img_mask, gt_mask)
    j = np.count_nonzero(img_and)
    i = np.count_nonzero(img_or)
    iou_value = float(float(j)/float(i))
    return iou_value


def overlap_calculator(img_mask, gt_mask):
    gt_mask *= 1.0
    img_and = cv2.bitwise_and(img_mask, gt_mask)
    j = np.count_nonzero(img_and)
    i = np.count_nonzero(gt_mask)
    overlap_value = float(float(j)/float(i))
    return overlap_value


'''
iou_iteration function calculates at each time step which is the ground truth object
that overlaps more with the visual region, so that we can calculate the rewards appropriately
'''


def iou_iteration(gt_masks, region_mask, objects, class_id, last_matrix, available_objects):
    results = np.zeros(len(objects))

    for i in range(len(objects)):
        if not objects[i] == class_id:
            continue

        if not available_objects[i] == 1:
            results[i] = -1  # Change the value here
        else:
            gt_mask = gt_masks[:, :, i]
            iou = iou_calculator(region_mask, gt_mask)
            results[i] = iou

    index = np.argmax(results)  # Which object has the most overlap value with current region.
    old_iou = last_matrix[index]
    # Record the former results, use a matrix because we may changed object during masking.
    new_iou = max(results)  # Record the new iou value

    # print(old_iou, new_iou, results, index)
    return old_iou, new_iou, results, index


