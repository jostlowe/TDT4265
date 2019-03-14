import numpy as np
import matplotlib.pyplot as plt
import json
import copy
from task2_tools import read_predicted_boxes, read_ground_truth_boxes


def calculate_iou(pred_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        pred_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """
    x_left = max(pred_box[0], gt_box[0])
    x_right = min(pred_box[2], gt_box[2])
    y_top = max(pred_box[1], gt_box[1])
    y_bottom = min(pred_box[3], gt_box[3])

    if x_left > x_right or y_top > y_bottom:
        return 0.0

    overlap = (x_right - x_left)*(y_bottom - y_top)

    pred_area = (pred_box[2]-pred_box[0])*(pred_box[3] - pred_box[1])
    gt_area = (gt_box[2]-gt_box[0])*(gt_box[3]-gt_box[1])
    union_area = pred_area + gt_area - overlap

    return overlap/float(union_area)


def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """
    if num_tp == num_fp:
        return 1
    return num_tp/float(num_tp+num_fp)


def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """
    if num_tp == num_fn:
        return 0
    return num_tp/float(num_tp+num_fn)

def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    """
    # Find all possible matches with a IoU >= iou threshold

    # Sort all matches on IoU in descending order

    # Find all matches with the highest IoU threshold
    match_pred = []
    match_gt = []

    for gt_box in gt_boxes:
        best_iou = 0
        best_pred_box = None

        for pred_box in prediction_boxes:
            iou = calculate_iou(pred_box, gt_box)
            if iou >= iou_threshold and iou > best_iou:
                best_pred_box = pred_box
                best_iou = iou

        # No matching pred.box
        if best_pred_box is not None:
            match_pred.append(best_pred_box)
            match_gt.append(gt_box)

    # Convert to numpy arrays
    return np.array(match_pred), np.array(match_gt)


def calculate_individual_image_result(
        prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, "false_neg": int}
    """
    pred_match, gt_match = get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold)

    num_true_pos = pred_match.shape[0]
    num_false_pos = prediction_boxes.shape[0] - num_true_pos
    num_false_neg = gt_boxes.shape[0] - num_true_pos
    return {"true_pos": num_true_pos, "false_pos": num_false_pos, "false_neg": num_false_neg}

def calculate_precision_recall_all_images(
        all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """
    total_true_pos, total_false_pos, total_false_neg = 0, 0, 0
    for image_pred_boxes, image_gt_boxes in zip(all_prediction_boxes, all_gt_boxes):
        num_dict = calculate_individual_image_result(image_pred_boxes, image_gt_boxes, iou_threshold)

        total_true_pos += num_dict["true_pos"]
        total_false_pos += num_dict["false_pos"]
        total_false_neg += num_dict["false_neg"]

    precision = calculate_precision(total_true_pos, total_false_pos, total_false_neg)
    recall = calculate_recall(total_true_pos, total_false_pos, total_false_neg)

    return precision, recall


def get_precision_recall_curve(all_prediction_boxes, all_gt_boxes,
                               confidence_scores, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the precision-recall curve over all images. Use the given
       confidence thresholds to find the precision-recall curve.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        tuple: (precision, recall). Both np.array of floats.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    # DO NOT CHANGE. If you change this, the tests will not pass when we run the final
    # evaluation
    confidence_thresholds = np.linspace(0, 1, 500)

    precision = []
    recall = []

    # Sort out predictions with confidence below threshold
    for c_t in confidence_thresholds:
      img_pred_array = []
      for img_num, pred_boxes in enumerate(all_prediction_boxes):
        pred_array = []
        for box_num, pred_box in enumerate(pred_boxes):

            if confidence_scores[img_num][box_num] >= c_t:
              pred_array.append(pred_box)

        pred_array = np.array(pred_array)
        img_pred_array.append(pred_array)

      img_pred_array = np.array(img_pred_array)

      prc, rcl = calculate_precision_recall_all_images(img_pred_array, all_gt_boxes, iou_threshold)

      precision.append(prc)
      recall.append(rcl)

    return (np.array(precision), np.array(recall))


def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    # No need to edit this code.
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    # Calculate the mean average precision given these recall levels.
    # DO NOT CHANGE. If you change this, the tests will not pass when we run the final
    # evaluation
    recall_levels = np.linspace(0, 1.0, 11)

    max_precision_list = []

    for level in recall_levels:
        max_precision = 0
        for precision, level_marked in zip(precisions, recalls):
            if level_marked >= level and precision >= max_precision:
                max_precision = precision
        max_precision_list.append(max_precision)
    return np.average(max_precision_list)


def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)
    iou_threshold = 0.5
    precisions, recalls = get_precision_recall_curve(all_prediction_boxes,
                                                     all_gt_boxes,
                                                     confidence_scores,
                                                     iou_threshold)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions,
                                                              recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))


if __name__ == "__main__":
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)
