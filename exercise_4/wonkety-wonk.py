import numpy as np
def calculate_iou(pred_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """
    xA = max(pred_box[0], gt_box[0])
    yA = max(pred_box[1], gt_box[1])
    xB = min(pred_box[2], gt_box[2])
    yB = min(pred_box[3], gt_box[3])

    internal_area = max(0, xB-xA+1)*max(0, yB-yA+1)

    pred_box_area = (pred_box[2] - pred_box[0] + 1) * (pred_box[3] - pred_box[1] + 1)
    gt_box_area = (gt_box[2]-gt_box[0]+1)*(gt_box[3]-gt_box[1]+1)

    return internal_area/float(pred_box_area + gt_box_area - internal_area)


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
    # calculate max ious for all prediction boxes
    max_iou_func = lambda pred_box: max([calculate_iou(pred_box, gt_box) for gt_box in gt_boxes])
    max_ious = np.apply_along_axis(max_iou_func, 1, prediction_boxes)

    num_boxes = prediction_boxes.shape[0]

    filtered_pred_boxes, filtered_gt_boxes, filtered_max_ious = [np.empty((0, 4))]*3
    print(filtered_gt_boxes)

    print(prediction_boxes, max_ious)







a = np.array([[1,1,3,3], [2,2,4,4], [1,2,4,5]])
b = np.array([[2,2,3,3], [2,2,4,4], [5,3,5,5]])
print(get_all_box_matches(a,b, 0.5))
