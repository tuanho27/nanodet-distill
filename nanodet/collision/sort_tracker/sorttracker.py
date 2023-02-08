"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
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
from .utils import *

class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox, shape=[1208, 1920], camera_fps=30, focal_length=2617, ttc_type='width', camera_height=1.4):
        """
        Initialises a tracker using initial bounding box.
        shape = [H, W] of image
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        self.kf_Tm = KalmanFilter(dim_x=2, dim_z=1)
        self.kf_Tm.F = np.array([
            [1, 1],
            [0, 1]
        ])
        self.kf_Tm.H = np.array([
            [1, 0]
        ])
        # self.kf_Tm.x = np.array([[20.],
        #  [0.]])
        self.kf_Tm.R *= 1.
        self.kf_Tm.P[1:, 1:] *= 10.
        self.kf_Tm.Q[1:, 1:] *= 0.01

        self.ttc_type = ttc_type
        self.camera_height = camera_height
        self.distance = 1.0
        self.previous_distance = None
        self.distance_rate = 0.0

        # init lateral position
        self.focal_length = focal_length
        self.shape = shape
        self.position = convert_bbox_to_lateral_position(
            bbox, shape)   # [x_l, x_r, x_r-x_l]
        # [[x_l, x_r, x_r-x_l], [x_l, x_r, x_r-x_l], ...]
        self.positions = deque(maxlen=9)
        self.positions.append(self.position)
        self.lateral_position = [0.0, 0.0]  # [X_l, X_r]
        self.emergency_vehicle = False
        self.left_line = (None, None)  # (A, B), f(t) = At + B
        self.right_line = (None, None)  # (A, B), f(t) = At + B
        # initialize TCC
        self.camera_fps = camera_fps
        self.delta_t = 1/camera_fps
        self.previous_width = bbox[2] - bbox[0]
        self.previous_height = bbox[3] - bbox[1]
        self.previous_Tm = 0
        self.ttc = 0
        self.previous_S = 0
        self.n = 0
        # for ttc testing
        self.previous_tm_test = 0.0
        self.ttc_no_kalman = 0.0
        self.estimate_TCC()

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(
            convert_bbox_to_z(bbox)
        )

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6]+self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1

        if(self.time_since_update > 0):
            self.hit_streak = 0

        self.time_since_update += 1
        self.history.append(
            convert_x_to_bbox(self.kf.x)
        )
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)

    def estimate_TCC(self, ) -> float:
        current_bbox = self.get_state()
        current_width = current_bbox[0, 2] - current_bbox[0, 0]
        current_height = current_bbox[0, 3] - current_bbox[0, 1]

        current_position = convert_bbox_to_lateral_position(
            current_bbox[0], self.shape)
        self.positions.append(current_position)

        self.emergency_vehicle = False
        if len(self.positions) == 9:
            is_emergency, self.left_line, self.right_line = extrapolate_lateral_position(
                self.positions, self.ttc, self.delta_t
            )
            # if self.ttc > 0 and self.ttc <= 2.5:
            self.emergency_vehicle = is_emergency

        self.estimate_range(current_bbox[0])
        self.estimate_range_rate(current_bbox[0], current_width)
        self.previous_width = current_width
        self.previous_height = current_height
        if self.ttc_type == 'width':
            S = current_width / self.previous_width
        elif self.ttc_type == 'area':
            S = math.sqrt((current_width * current_height) /
                          (self.previous_width * self.previous_height))
        else:
            raise("TTC evaluation type must be 'width' or 'area'")

        if self.previous_distance is None:
            self.ttc = self.distance/5.0  # Maximum TTC
        else:
            m = max(self.previous_distance, self.distance)
            if m == 0:
                m = 1e-3
            self.ttc = self.ttc*(1-(self.previous_distance-self.distance)/m)
        # print(self.ttc)
        self.ttc = np.clip(self.ttc, 0, 10)  # Maximum TTC

        self.previous_distance = self.distance

    def estimate_range(self, d):
        center = int(self.shape[0]/2) + 100  # adjust the center
        y = abs(center - d[3])
        self.distance = self.focal_length * self.camera_height / y

    def estimate_range_rate(self, d, current_width):
        s = (current_width - self.previous_width) / self.previous_width
        self.distance_rate = self.distance * s / self.delta_t


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if(len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if(len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class SORT(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.8, shape=[1208, 1920], camera_fps=30., ttc_type='width',  focal_length=2617, camera_height=1.4):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.trackers: List[KalmanBoxTracker]
        self.frame_count = 0
        self.camera_fps = camera_fps
        self.shape = shape
        self.ttc_type = ttc_type
        self.focal_length = focal_length
        self.camera_height = camera_height

    def update(self, dets=np.empty((0, 5))) -> List[KalmanBoxTracker]:
        """
        Params:
        dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unconfirm_tracks = associate_detections_to_trackers(
            dets, trks, self.iou_threshold
        )
        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(
                dets[i, :], shape=self.shape, camera_fps=self.camera_fps, ttc_type=self.ttc_type,
                focal_length=self.focal_length, camera_height=self.camera_height)
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if(trk.time_since_update <= self.max_age):
                trk.estimate_TCC()

            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(trk)

            i -= 1
            # remove dead tracklet
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        return ret

