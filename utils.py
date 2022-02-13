from cv2 import accumulate
import numpy as np
import torch

def iou(pred, target, n_classes = 10):
  ious = []
  pred = pred.view(-1)
  target = target.view(-1)

  # Ignore IoU for undefined class ("9")
  for cls in range(n_classes-1):  # last class is ignored
    pred_inds = pred == cls
    target_inds = target == cls
    intersection = (pred_inds[target_inds]).sum().data  # Cast to long to prevent overflows
    union = pred_inds.sum().data + target_inds.sum().data - intersection
    if union == 0:
      ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
    else:
      ious.append(float(intersection) / float(max(union, 1)))
  return np.array(ious)

def pixel_acc(pred, target):
    #TODO complete this function, make sure you don't calculate the accuracy for undefined class ("9")
    total = torch.sum(target < 9).item()
    correct = torch.sum((pred == target) * (target < 9)).item()
    # total = target.size(0) * target.size(1) * target.size(2)
    # correct = (pred == target).sum().item()
    acc = correct / total
    return acc 