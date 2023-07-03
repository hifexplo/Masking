import cv2
import sys
import torch
import os

import numpy as np
import matplotlib as mpl
import pandas as pd

import torchvision.transforms as T
import torch.utils.data as dataf
import torch.nn as nn
import matplotlib.pyplot as plt


from sklearn.metrics import confusion_matrix
from spectral import *


def show_anns(anns):
    """
    Special function provided by SAM in order to visualize the results
    For further information, check automatic mask generator example:
    https://github.com/facebookresearch/segment-anything/tree/main/notebooks
    """
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))


def object_coordination_extraction(box, input_image):
    """
    Extract the coordination from dino's box prediction.
    
    Args:
        predictions (torch.Tensor): The bounding boxes of dino's detected
                                    objects.
                                    
        input_image (numpy):        the input image.

    Returns:
        X, Y (int): bounding box center coords.
        W, H (int): bounding box width, height.
    """
    
    # Extract biggest object bbox coords
    X = int(box[0].item()*input_image.shape[1])
    Y = int(box[1].item()*input_image.shape[0])
    W = int(box[2].item()*input_image.shape[1])
    H = int(box[3].item()*input_image.shape[0])
    
    return X, Y, W, H

def intersection_filtering(masks,boxes,img, C =5, Max_area = 15000, Center = True):
    """
    Takes the output of SAM (list of segmentation masks) and gives
    back a list that contains masks that are only INSIDE Grounding
    dino's bounding box.
    
    The conditions for the masks to stay are:
    
    1. to be inside the Dino's predicted boxes.
    2. the area of the mask is smaller than Max_area (int).
    
    Args:
        masks    (list):         SAM's predictions.
        boxes    (torch.Tensor): The bounding boxes of dino's detected objects.
        C        (int)         : Edge buffer in pixels for the edges of the
                                 boxes.
        Max_area (int)         : the miximum area of the allowed mask in pixel.
        Center   (bool)        : If true: evaluate based on the center coordinates
                                 else: evaluate based on the Top-Left coordinates.
    
    Returns:
        masks_to_keep (list)   : A list of the filtered masks.
        
    """
    # Define empty list for the final filtered masks
    masks_to_keep = []
    
    # Extract Grounding dino's biggest object BBox
    # boxes[0] is the biggest predicted box by Dino
    for box in boxes:
        X, Y, W, H = object_coordination_extraction(box, img)
        
        
    # Loop through SAM's masks
    for i in range(len(masks)):

        # Extract the coors of each mask
        x,y,w,h = masks[i]["bbox"]
        
        
        # Fiter based on Center coords or Top-Left coords
        if Center:
            x = x + int(w/2)
            y = y + int(h/2)
        
        # Extract the area of the mask
        area = masks[i]["area"]
        
        # Intersection filtering
        mask_cond_x_1 = x < (X + int(W/2) - C)
        mask_cond_x_2 = x > (X - int(W/2) + C)
        mask_cond_x = mask_cond_x_1 and mask_cond_x_2
        mask_cond_y_1 = y < (Y + int(H/2) - C)
        mask_cond_y_2 = y > (Y - int(H/2) + C)
        mask_cond_y = mask_cond_y_1 and mask_cond_y_2
    
        if mask_cond_x and mask_cond_y and area < Max_area:
            masks_to_keep.append(masks[i])
            
    return masks_to_keep

def excluding_filtering(masks,boxes,img, C =5, Max_area = 500, Center = True):
    """
    Takes the output of SAM (list of segmentation masks) and gives
    back a list that contains masks that are only OUTSIDE Grounding
    dino's bounding box.

    The conditions for the masks to stay are:
    
    1. to be outside the Dino's predicted boxes.
    2. the area of the mask is smaller than Max_area (int).
    
    Args:
        masks    (list):         SAM's predictions.
        boxes    (torch.Tensor): The bounding boxes of dino's detected objects.
        C        (int)         : Edge buffer in pixels for the edges of the
                                 boxes.
        Max_area (int)         : the miximum area of the allowed mask in pixel.
        Center   (bool)        : If true: evaluate based on the center coordinates
                                 else: evaluate based on the Top-Left coordinates.
    
    Returns:
        masks_to_keep (list)   : A list of the filtered masks.
        
    """
    # Define empty list for the final filtered masks
    masks_to_keep = []
        
    # Loop through SAM's masks
    for i in range(len(masks)):

        # Extract the coors of each mask
        x,y,w,h = masks[i]["bbox"]
        
        # Fiter based on Center coords or Top-Left coords
        if Center:
            x = x + int(w/2)
            y = y + int(h/2)
        
        # Extract the area of the mask
        area = masks[i]["area"]
        
        Flag = True
        
        # Extract Grounding dino's biggest object BBox
        # boxes[-1] is the biggest predicted box by Dino
        for box in boxes:
            # Extract Dino's boxes coords
            X, Y, W, H = object_coordination_extraction(box, img)
        
            # Excluding filtering
            mask_cond_x_1 = x < (X + int(W/2) + C)
            mask_cond_x_2 = x > (X - int(W/2) - C)
            mask_cond_x = mask_cond_x_1 and mask_cond_x_2
            mask_cond_y_1 = y < (Y + int(H/2) + C)
            mask_cond_y_2 = y > (Y - int(H/2) - C)
            mask_cond_y = mask_cond_y_1 and mask_cond_y_2
            
            if mask_cond_x and mask_cond_y or area > Max_area:
                Flag = False
        
        if Flag:
            masks_to_keep.append(masks[i])
                
            
    return masks_to_keep

def mask_generation(filtered_masks):
    """
    Build the foreground mask from the list of the final filtered masks.
    
    Args:
        filtered_masks (list): list of SAM's filtered masks.

    Retruns:
        output (numpy): The final mask.
    """
    
    # Get the shape of the input image
    shape = filtered_masks[0]['segmentation'].shape
    
    # Create an empty image with the same shape as the input image
    output = np.zeros(shape, dtype=bool)
    
    # Loop over all the filtered masks in the list and union them
    # in the output image
    for i, seg in enumerate (filtered_masks):
        output |= seg['segmentation']
        
    return output


def evaluate_segmentation(ground_truth_masks, predicted_masks, num_classes):
    """
    Evaluates the performance of a semantic segmentation model
    by calculating various evaluation metrics.

    Args:
        ground_truth_masks (list): List of ground truth segmentation masks.
        predicted_masks    (list): List of predicted segmentation masks.
        num_classes        (int): Number of classes in the segmentation task.

    Returns:
        tuple: A tuple containing the following evaluation metrics:
            - confusion_matrix_sum (numpy.ndarray): Summed confusion matrix.
            - true_positive_sum    (numpy.ndarray): Sum of true positives 
                                                    per class.
            - true_negative_sum    (numpy.ndarray): Sum of true negatives per 
                                                    class (always zero for semantic
                                                    segmentation).
            - false_positive_sum   (numpy.ndarray): Sum of false positives per
                                                    class.
            - false_negative_sum   (numpy.ndarray): Sum of false negatives per
                                                    class.
            - precision            (numpy.ndarray): Precision per class.
            - recall               (numpy.ndarray): Recall per class.
            - f1_score             (numpy.ndarray): F1 score per class.
            - pixel_accuracy_per_class (numpy.ndarray): Pixel accuracy per
                                                        class.
            - pixel_accuracy       (float)        : Overall pixel accuracy.
            - iou                  (numpy.ndarray): Intersection over Union (IoU)
                                                    per class.
            - dice_coefficient     (numpy.ndarray): Dice coefficient per class.
            - kappa                (float)        : Kappa coefficient.

    """
    # Initialize variables for aggregating evaluation metrics
    confusion_matrix_sum = np.zeros((num_classes, num_classes), dtype=np.int64)
    true_positive_sum = np.zeros(num_classes, dtype=np.int64)
    true_negative_sum = np.zeros(num_classes, dtype=np.int64)
    false_positive_sum = np.zeros(num_classes, dtype=np.int64)
    false_negative_sum = np.zeros(num_classes, dtype=np.int64)
    intersection_sum = np.zeros(num_classes, dtype=np.int64)
    union_sum = np.zeros(num_classes, dtype=np.int64)

    for gt_mask, pred_mask in zip(ground_truth_masks, predicted_masks):
        # Calculate confusion matrix
        cm = confusion_matrix(gt_mask.flatten(), pred_mask.flatten(), labels=list(range(num_classes)))
        confusion_matrix_sum += cm

        # Calculate true positive, true negative, false positive, false negative
        true_positive = np.diag(cm)
        true_positive_sum += true_positive

        false_positive = np.sum(cm, axis=0) - true_positive
        false_positive_sum += false_positive

        false_negative = np.sum(cm, axis=1) - true_positive
        false_negative_sum += false_negative

        # Calculate intersection and union for Intersection Over Union (IoU)
        intersection = true_positive
        union = np.sum(cm, axis=1) + np.sum(cm, axis=0) - true_positive
        intersection_sum += intersection
        union_sum += union

    # Calculate pixel accuracy per class
    pixel_accuracy_per_class = true_positive_sum / (true_positive_sum + false_negative_sum)

    # Calculate pixel accuracy
    pixel_accuracy = np.sum(true_positive_sum) / np.sum(confusion_matrix_sum)

    # Calculate precision, recall, F1 score
    precision = true_positive_sum / (true_positive_sum + false_positive_sum)
    recall = true_positive_sum / (true_positive_sum + false_negative_sum)
    f1_score = (2 * precision * recall) / (precision + recall)

    # Calculate Intersection Over Union (IoU)
    iou = intersection_sum / union_sum

    # Calculate Dice coefficient
    dice_coefficient = (2 * intersection_sum) / (np.sum(confusion_matrix_sum, axis=1) + \
                                                 np.sum(confusion_matrix_sum, axis=0))

    # Calculate Kappa coefficient
    total_pixels = np.sum(confusion_matrix_sum)
    observed_accuracy = np.sum(true_positive_sum) / total_pixels
    expected_accuracy = np.sum(true_positive_sum) * np.sum(confusion_matrix_sum, axis=1) / total_pixels**2
    kappa = (observed_accuracy - expected_accuracy) / (1 - expected_accuracy)

    # Return the calculated evaluation metrics
    return confusion_matrix_sum, true_positive_sum, true_negative_sum, false_positive_sum, false_negative_sum,\
           precision, recall, f1_score, pixel_accuracy_per_class,   pixel_accuracy, iou, dice_coefficient,\
           kappa
