import argparse
import json
import os

import cv2
import numpy as np
import torch

from matplotlib import pyplot as plt
from numpy.core.einsumfunc import _parse_possible_contraction
from numpy.testing._private.utils import measure
from tqdm import tqdm
from tqdm.std import TRLock

from nanodet.collision.trackers.sorttracker import SORT, KalmanBoxTracker
from nanodet.collision.timer import Timer
from collections import defaultdict

classes = ['person', 'car','motorbike']


_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
    ]
).astype(np.float32).reshape(-1, 3)


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color

def plot_det(img, boxes, color=None):
    for i in range(len(boxes)):
        box = boxes[i][:4]
        score =  boxes[i][4]
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])
        if color is None:
            color = (255,0,0)
        else:
            color = color
        text = '{:.1f}%'.format(score * 100)
        txt_color = (255, 0, 0) #if np.mean(_COLORS[0]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img


def plot_tracking(image, tlwhs, obj_ids, cls_ids, vels=None, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255
    text_scale = 2
    text_thickness = 2
    line_thickness = 3
    blur_mode = 0
    radius = max(5, int(im_w/140.))
    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)
    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        cat_id = str(classes[int(cls_ids[i])])
        vel = round(vels[i],2)
        cv2.putText(im, cat_id, (intbox[0]+15, intbox[1] + 15),
                    cv2.FONT_HERSHEY_PLAIN, 1, (10, 20, 255), thickness=2)
        cv2.putText(im, f'R Vel: {vel}', (intbox[0]+15, intbox[1] + 30),
                    cv2.FONT_HERSHEY_PLAIN, 1, (10, 200, 255), thickness=2)
        cv2.putText(im, f'Dis: {0}', (intbox[0]+15, intbox[1] + 45),
                    cv2.FONT_HERSHEY_PLAIN, 1, (10, 200, 255), thickness=2)
        if blur_mode == 1:
            im[intbox[1]:intbox[3], intbox[0]:intbox[2]] = cv2.GaussianBlur(im[intbox[1]:intbox[3], intbox[0]:intbox[2]], (15, 15), 0)

        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        color = get_color(abs(obj_id))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)
    return im


def inCollisionArea(contour: np.ndarray,
                    bbox: np.ndarray):
    left_point = (bbox[0], bbox[3])
    right_point = (bbox[2], bbox[3])
    center_point = ((bbox[0] + bbox[2])//2, bbox[3])
    return cv2.pointPolygonTest(contour, left_point, False) >= 0 or \
        cv2.pointPolygonTest(contour, right_point, False) >= 0 or \
        cv2.pointPolygonTest(contour, center_point, False) >= 0

# def drawGuideLines(im: np.ndarray,
#                    contour: np.ndarray,
#                    style: str = 'dotted',
#                    gap: int = 20):
#     '''
#     contour: guidelines triangle np.array([[[0,0],[1,1],[2,2]]])
#             0
#           /   \
#          1 ___ 2
#     '''
#     thickness = 3
#     color = (0, 0, 0)
#     if style == 'dotted':
#         edge = ((contour[0, 0, 0] - contour[0, 1, 0])**2 +
#                 (contour[0, 0, 1] - contour[0, 1, 1])**2)**.5

#         for i in range(0, int(edge), gap):
#             r = i/edge
#             xl = int((contour[0, 0, 0]*(1-r) + contour[0, 1, 0]*r)+0.5)
#             xr = int((contour[0, 0, 0]*(1-r) + contour[0, 2, 0]*r)+0.5)
#             yl = int((contour[0, 0, 1]*(1-r) + contour[0, 1, 1]*r)+0.5)
#             pl = (xl, yl)
#             pr = (xr, yl)
#             cv2.circle(im, pl, thickness, color)
#             cv2.circle(im, pr, thickness, color)
#             color = (color[0] + 30, color[1], color[2]
#                      ) if color[0] < 225 else color
#     else:
#         im = cv2.polylines(img=im, pts=contour, isClosed=True,
#                            color=(255, 0, 0), thickness=3)
#     return im

def drawGuideLines(im: np.ndarray,
                   contour: np.ndarray):
    #300-500cm
    im = cv2.line(im, (contour[4][0],contour[4,1]),(contour[3][0],contour[3,1]), (0,0,255),5)
    im = cv2.line(im, (contour[4][2],contour[4,3]),(contour[3][2],contour[3,3]), (0,0,255),5)
    #500-100
    im = cv2.line(im, (contour[3][0],contour[3,1]),(contour[2][0],contour[2,1]), (0,255,255),5)
    im = cv2.line(im, (contour[3][2],contour[3,3]),(contour[2][2],contour[2,3]), (0,255,255),5)
    #1000-2000
    im = cv2.line(im, (contour[2][0],contour[2,1]),(contour[1][0],contour[1,1]), (0,255,0),5)
    im = cv2.line(im, (contour[2][2],contour[2,3]),(contour[1][2],contour[1,3]), (0,255,0),5)
    #2000-3000
    im = cv2.line(im, (contour[1][0],contour[1,1]),(contour[0][0],contour[0,1]), (0,255,10),5)
    im = cv2.line(im, (contour[1][2],contour[1,3]),(contour[0][2],contour[0,3]), (0,255,10),5)

    # |-  -|
    im = cv2.line(im, (contour[0][0],contour[0,1]),(contour[0][0]+20,contour[0,1]), (0,255,10),5)
    im = cv2.line(im, (contour[0][2],contour[0,3]),(contour[0][2]-20,contour[0,3]), (0,255,10),5)

    cw_track_region = np.array(contour[:5,:4]).reshape(10,2)
    # im = cv2.drawContours(im, cw_track_region.reshape(-1,1,2), -1, color= (20,150,20), thickness=-1)
    im = cv2.polylines(img=im, pts=cw_track_region.reshape(-1,1,2), isClosed=True, color=(255, 0, 0), thickness=6)
    
    return im

def visualize(im: np.ndarray,
           trk: KalmanBoxTracker,
           params: argparse.ArgumentParser,
           contour: np.ndarray):

    bbox = trk.get_state()[0]
    TTC = trk.ttc
    Tm = trk.previous_Tm
    isEmergency = False
    distance = trk.distance
    distance_rate = trk.distance_rate
    t = distance / distance_rate if distance_rate != 0.0 else 100.0

    left_point = (bbox[0], bbox[3])
    right_point = (bbox[2], bbox[3])
    center_point = ((bbox[0] + bbox[2])//2, bbox[3])

    # If vehicle get inside the guidelines area
    if inCollisionArea(contour, bbox):
        # if it's going to contact: red
        if TTC > 0 and TTC <= params.ttc_warning and isEmergency and t > 0 and t < params.ttc_warning:
            color = (0, 0, 255)
            border = 5
            isEmergency = True
        # if emergency vehicle but not going to contact: yellow
        else:
            color = (0, 255, 255)
            border = 2
    # others: green
    else:
        color = (10, 255, 0)
        border = 2

    # centroid = int(bbox[0] + bbox[2])//2, int(bbox[1] + bbox[3])//2
    cv2.rectangle(
        im,
        (int(bbox[0]), int(bbox[1])),
        (int(bbox[2]), int(bbox[3])),
        color, border
    )

    # Show TTC for object same lane, TTC <= ttc_threshold
    cv2.putText(
        im,
        f'TTC:{TTC:.1f}',
        (int(bbox[0]), int(bbox[3]) - 15),
        cv2.FONT_HERSHEY_PLAIN,
        # cv2.FONT_HERSHEY_SIMPLEX,
        1, (255, 20, 10), 2
    )

    if len(trk.positions) < 9:
        return

    if params.debug:
        # drawing for lateral
        cv2.line(im, (im.shape[1]//2, 0), (im.shape[1] //
                 2, im.shape[0]), (255, 255, 255), thickness=1)

        positions = np.array(trk.positions, dtype=np.float32)
        scale = (bbox[3] - bbox[1]) / im.shape[1]
        # positions *= scale

        delta_t_in_pixel = 10
        # start_left = (0-trk.left_line[1])/trk.left_line[0] # t = Ax + B
        # end_left = (trk.delta_t*(len(positions)) - trk.left_line[1])/trk.left_line[0] # t = Ax + B
        end_left = trk.left_line[0]*0 + trk.left_line[1]  # x = At + B
        # x = At + B
        start_left = trk.left_line[0] * \
            (TTC+trk.delta_t*(len(positions)-1)) + trk.left_line[1]
        # start_left *= scale
        # end_left *= scale
        cv2.line(
            im,
            (int(start_left + im.shape[1]/2),
             int(bbox[3] - TTC/trk.delta_t*delta_t_in_pixel)),
            (int(end_left + im.shape[1]/2), int(bbox[3]
                                                ) + delta_t_in_pixel*(len(positions))),
            (0, 255, 0), thickness=2
        )
        # start_right = (0-trk.right_line[1])/trk.right_line[0] # t = Ax + B
        # end_right = (trk.delta_t*(len(positions)) - trk.right_line[1])/trk.right_line[0] #t = Ax + B
        end_right = trk.right_line[0]*0 + trk.right_line[1]  # x = At + B
        # x = At + B
        start_right = trk.right_line[0] * \
            (TTC+trk.delta_t*(len(positions)-1)) + trk.right_line[1]

        # start_right *= scale
        # end_right *= scale
        cv2.line(
            im,
            (int(start_right + im.shape[1]/2),
             int(bbox[3] - TTC/trk.delta_t*delta_t_in_pixel)),
            (int(end_right + im.shape[1]/2), int(bbox[3]
                                                 ) + delta_t_in_pixel*(len(positions))),
            (0, 255, 0), thickness=2
        )

        for i, pos in enumerate(positions[::-1]):
            cv2.circle(
                im,
                (int(pos[0]+im.shape[1]/2), int(bbox[3]) + i*delta_t_in_pixel),
                3, (255, 0, 0), -1
            )
            cv2.circle(
                im,
                (int(pos[1]+im.shape[1]/2), int(bbox[3]) + i*delta_t_in_pixel),
                3, (0, 0, 255), -1
            )


def obj2dict(preds, trackers):
    labels = []
    for i, trk in enumerate(trackers):
        bbox = trk.get_state()[0]
        obj = {}
        cls = int(preds[i][5])
        if cls not in classes.keys():
            continue
        obj['category'] = classes[cls]
        obj['id'] = trk.id
        obj['box2d'] = {
            'x1': bbox[0],
            'y1': bbox[1],
            'x2': bbox[2],
            'y2': bbox[3]
        }
        obj['score'] = float(preds[i][4].cpu().numpy())
        labels.append(obj)
    return labels


def coords(s):
    try:
        x, y = map(int, s.split(','))
        return x, y
    except:
        raise argparse.ArgumentTypeError("Coordinates must be x,y")
