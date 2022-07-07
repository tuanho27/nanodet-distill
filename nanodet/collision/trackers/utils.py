from __future__ import print_function

import os
import argparse
import posixpath
import math
from operator import pos
from typing import List
from collections import deque
from cv2 import sqrt

import numpy as np
from filterpy.kalman import KalmanFilter
from cython_bbox import bbox_overlaps as bbox_ious
from scipy.spatial.distance import cdist

np.random.seed(0)


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

def euclidean_distance(atracks, btracks, metric='euclidean'):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]
    :rtype cost_matrix np.ndarray
    """
    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [convert_z_to_bbox(track['det_info'][-1]['pred_state'])for track in atracks]
        btlbrs = [track['roi'] for track in btracks]
    l2_dist = np.ones((len(atlbrs), len(btlbrs)), dtype=np.float)*1000
    for id_track, corr_track in enumerate(atlbrs):
        for id_det, corr_det in enumerate(btlbrs):
            try:
                l2_dist[id_track, id_det] = cdist(np.array([corr_track]), np.array([corr_det]), metric)
            except:
                l2_dist[id_track, id_det]=1000
    return l2_dist

def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )

    return ious


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """
    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [convert_z_to_bbox(track['det_info'][-1]['pred_state'])for track in atracks]
        btlbrs = [track['roi'] for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def v_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in atracks]
        btlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def convert_z_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  try:
      w = np.sqrt(x[2]*x[3])
  except:
      pass
  h = x[2]/w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((4, 1))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((5, 1))

############

def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return(o)


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
        [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
        the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
        [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if(score == None):
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
    else:
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))


def convert_bbox_to_lateral_position(bbox, shape):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and image size [H, W] and returns lateral_position in the form
        [x_l, x_r, x_r-x_l] where x_r, x_l are image coordinates of the left and right egdes of target
        vehicle
    """
    x_l = bbox[0] - shape[1]/2          # x - W
    x_r = x_l + bbox[2] - bbox[0]       # x_l + w
    p = x_r - x_l

    return np.array([x_l, x_r, p])


def extrapolate_lateral_position(positions, ttc, delta_t):
    """
    positions: deque of size 9, each element is a array of [x_l, x_r, x_r-x_l]
    ttc: float

    Perform a linear fit to find X_l and X_r line, then extrapolate them to the time t = ttc
    Return True if the target vehicle is on collision course, False otherwise
    """
    pos = np.array(positions)
    t = np.arange(len(pos)) * delta_t
    t = np.vstack([t, np.ones(len(pos))]).T
    a_Xl, b_Xl = np.linalg.lstsq(t, pos[:, 0], rcond=None)[0]
    a_Xr, b_Xr = np.linalg.lstsq(t, pos[:, 1], rcond=None)[0]

    X_l_ttc = a_Xl * (ttc + delta_t*(len(pos)-1)) + b_Xl
    X_r_ttc = a_Xr * (ttc + delta_t*(len(pos)-1)) + b_Xr

    return X_l_ttc * X_r_ttc <= 0, (a_Xl, b_Xl), (a_Xr, b_Xr)


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost