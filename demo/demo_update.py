import argparse
from ipaddress import ip_address
import os
import time
import cv2
from urllib3 import Retry
import torch
import sys
sys.path.append('./')
from tqdm import tqdm
from glob import glob
import imageio
import numpy as np
from shapely.geometry import Polygon
from shapely.validation import make_valid
from scipy.spatial import distance
from ensemble_boxes import weighted_boxes_fusion
from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from nanodet.data.transform import Pipeline
from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight
from nanodet.util.path import mkdir

from nanodet.collision.trackers.sorttracker import SORT, KalmanBoxTracker
from nanodet.collision.trackers.utils import *
from nanodet.collision.timer import Timer
from nanodet.collision.visualize import *

image_ext = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
video_ext = ["mp4", "mov", "avi", "mkv"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("--config", help="model config file path")
    parser.add_argument("--model", help="model file path")
    parser.add_argument("--path", default="./demo", help="path to images or video")
    parser.add_argument("--track", action='store_true', help="Add tracking by default")
    parser.add_argument("--min_box_area", default=100, type=float, help="min box area to eliminate noise")
    
    parser.add_argument('-fps', type=float,
                        help='Video FPS', default=10.0)
    parser.add_argument('-ttc_threshold', type=float,
                        help='Threshold for TTC warning', default=3.0)
    parser.add_argument('-ttc_warning', type=float,
                        help='Threshold for TTC warning', default=1.0)
    parser.add_argument('-ttc_type', type=str,
                        help='width/area', default='width')
    # Tracker
    parser.add_argument("-max_age",
                        help="Maximum number of frames to keep alive a track without associated detections.",
                        type=int, default=3)
    parser.add_argument("-min_hits",
                        help="Minimum number of associated detections before track is initialised.",
                        type=int, default=3)
    parser.add_argument("-iou_threshold",
                        help="Minimum IOU for match.", type=float, default=0.3)

    parser.add_argument("-debug", action='store_true',
                        help="Minimum IOU for match.")

    # Range
    parser.add_argument("-focal_length",
                        help="camera focal length in pixels", type=int, default=2617)
    parser.add_argument("-camera_height",
                        help="count from the camera to road surface in meters", type=float, default=1.4)
    parser.add_argument("-guidelines",
                        help="guidelines for camera, Ex: -guidelines 660,1208 840,1020 ...", type=coords, nargs=4, default=[[960, 820], [660, 1208], [1260, 1208]])

    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )
    args = parser.parse_args()
    return args


class Predictor(object):
    def __init__(self, cfg, model_path, logger, device="cuda:0"):
        self.cfg = cfg
        self.device = device
        model = build_model(cfg.model)
        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        load_model_weight(model, ckpt, logger)
        
        if cfg.model.arch.backbone.name == "RepVGG":
            deploy_config = cfg.model
            deploy_config.arch.backbone.update({"deploy": True})
            deploy_model = build_model(deploy_config)
            from nanodet.model.backbone.repvgg import repvgg_det_model_convert

            model = repvgg_det_model_convert(model, deploy_model)
        self.model = model.to(device).eval()
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        meta = dict(img_info=img_info, raw_img=img, img=img)
        meta = self.pipeline(None, meta, self.cfg.data.val.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(self.device)
        meta = naive_collate([meta])
        meta["img"] = stack_batch_img(meta["img"], divisible=32)
        with torch.no_grad():
            results = self.model.inference(meta)
        return meta, results

    def visualize(self, dets, meta, class_names, score_thres, wait=0):
        time1 = time.time()
        result_img = self.model.head.show_result(
            meta["raw_img"][0], dets, class_names, score_thres=score_thres, show=False
        )
        print("viz time: {:.3f}s".format(time.time() - time1))
        return result_img

### helper ###
def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in image_ext:
                image_names.append(apath)
    return image_names

def generator():
  while True:
    yield

def get_track_region(type):
    if type == "front" or type== "rear":
        # return np.array([300,230,750,1050]).astype(int)
        return np.array([230,300,1050,750]).astype(int)
    else:
        # return np.array([150,200,700,1080]).astype(int)
        return np.array([200,150,1080,700]).astype(int)

def read_guideline(path):
    return np.loadtxt(path)

def polyiou_overlap(polygon1, polygon2):
    polygon1 = make_valid(polygon1)
    polygon2 = make_valid(polygon2)
    intersect = polygon1.intersection(polygon2).area
    union = polygon1.union(polygon2).area
    iou = intersect / union
    return iou

def rect2poly(rect):
    w = rect[2] - rect[0]
    h = rect[3] - rect[1]
    return np.array([
                    [rect[0]   , rect[1]],
                    [rect[0]+w , rect[1]],
                    [rect[0]   , rect[1]+h],
                    [rect[2]   , rect[3]],
                    ])

def wbf(boxes, W, H):
    '''This function for re-nms the double bboxes after infer'''
    weights = [2, 1]
    iou_thr = 0.5
    skip_box_thr = 0.0001
    sigma = 0.1
    bboxes = boxes[:,:4]
    bboxes[:,0] = bboxes[:,0]/W
    bboxes[:,1] = bboxes[:,1]/H
    bboxes[:,2] = bboxes[:,2]/W
    bboxes[:,3] = bboxes[:,3]/H
    scores = boxes[:,4]
    labels = np.ones_like(scores) #no need
    oboxes, oscores, olabels = weighted_boxes_fusion([bboxes.tolist()], 
                                                  [scores.tolist()], 
                                                  [labels.tolist()], 
                                                  weights=None, 
                                                  iou_thr=iou_thr, 
                                                  skip_box_thr=skip_box_thr)
    oboxes[:,0] = oboxes[:,0]*W
    oboxes[:,1] = oboxes[:,1]*H
    oboxes[:,2] = oboxes[:,2]*W
    oboxes[:,3] = oboxes[:,3]*H
    return np.hstack([oboxes,oscores[:,None]])

# Define Infinite (Using INT_MAX 
# caused overflow problems)
INT_MAX = 10000
 
# Given three collinear points p, q, r, 
# the function checks if point q lies
# on line segment 'pr'
def onSegment(p:tuple, q:tuple, r:tuple) -> bool:
     
    if ((q[0] <= max(p[0], r[0])) &
        (q[0] >= min(p[0], r[0])) &
        (q[1] <= max(p[1], r[1])) &
        (q[1] >= min(p[1], r[1]))):
        return True
         
    return False
 
# To find orientation of ordered triplet (p, q, r).
# The function returns following values
# 0 --> p, q and r are collinear
# 1 --> Clockwise
# 2 --> Counterclockwise
def orientation(p:tuple, q:tuple, r:tuple) -> int:
     
    val = (((q[1] - p[1]) *
            (r[0] - q[0])) -
           ((q[0] - p[0]) *
            (r[1] - q[1])))
            
    if val == 0:
        return 0
    if val > 0:
        return 1 # Collinear
    else:
        return 2 # Clock or counterclock
 
def doIntersect(p1, q1, p2, q2):
     
    # Find the four orientations needed for 
    # general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
 
    # General case
    if (o1 != o2) and (o3 != o4):
        return True
     
    # Special Cases
    # p1, q1 and p2 are collinear and
    # p2 lies on segment p1q1
    if (o1 == 0) and (onSegment(p1, p2, q1)):
        return True
 
    # p1, q1 and p2 are collinear and
    # q2 lies on segment p1q1
    if (o2 == 0) and (onSegment(p1, q2, q1)):
        return True
 
    # p2, q2 and p1 are collinear and
    # p1 lies on segment p2q2
    if (o3 == 0) and (onSegment(p2, p1, q2)):
        return True
 
    # p2, q2 and q1 are collinear and
    # q1 lies on segment p2q2
    if (o4 == 0) and (onSegment(p2, q1, q2)):
        return True
 
    return False
 
# Returns true if the point p lies 
# inside the polygon[] with n vertices
def is_inside_polygon(points:list, bboxes:list) -> bool:     
    insides = []
    for p in bboxes:
        n = len(points)
        
        # There must be at least 3 vertices
        # in polygon
        if n < 3:
            return False
            
        # Create a point for line segment
        # from p to infinite
        extreme = (INT_MAX, p[1])
        count = i = 0
        
        while True:
            next = (i + 1) % n
            
            # Check if the line segment from 'p' to 
            # 'extreme' intersects with the line 
            # segment from 'polygon[i]' to 'polygon[next]'
            if (doIntersect(points[i],
                            points[next],
                            p, extreme)):
                                
                # If the point 'p' is collinear with line 
                # segment 'i-next', then check if it lies 
                # on segment. If it lies, return true, otherwise false
                if orientation(points[i], p,
                            points[next]) == 0:
                    return onSegment(points[i], p,
                                    points[next])
                                    
                count += 1
                
            i = next
            
            if (i == 0):
                break
            
        # Return true if count is odd, false otherwise
        insides.append(count % 2 == 1)
    return insides

def points2check(box):
    w = box[2] - box[0]
    h = box[3] - box[1]
    return [
            #br
            (box[2],box[3]),
            #bl
            (box[0],box[3]),
            #bm
            (box[2]-w/2, box[3]),
            #br - 1/4*w
            (box[2]-w/4, box[3]),
            #bl + 1/4*w
            (box[2]-3*w/4, box[3]),
            #br - 1/6*h
            # (box[2], box[3]-10),
            #bl - 1/6*h
            # (box[0], box[3]-10)
    ]

def main():
    args = parse_args()
    local_rank = 0
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    load_config(cfg, args.config)
    logger = Logger(local_rank, use_tensorboard=False)
    predictor = Predictor(cfg, args.model, logger, device="cuda:0")
    logger.log('Press "Esc", "q" or "Q" to exit.')
    current_time = time.localtime()
    # static_guide_points = read_guideline('./rear_view_guideline_calib.txt')
    #leftPointX leftPointY rightPointX rightPointY realWidth realDistance
    ## e34
    # static_guide_points = np.array([[526, 258, 743, 257, 2192, 3000],
    #                                 [485, 288, 784, 286, 2192, 2000],
    #                                 [402, 356, 866, 352, 2192, 1000],
    #                                 [330, 427, 939, 422, 2192, 500],
    #                                 [294, 468, 976, 463, 2192, 300],
    #                                 ], dtype=np.int32)
    ## fordedge
    #rear
    static_guide_points = np.array([[498,  455, 769,  456, 2578, 3000],
                                     [432, 483, 833,  484, 2578, 2000],
                                     [285, 546, 977,  549, 2578, 1000],
                                     [165, 598, 1099, 603, 2578, 500],
                                     [115, 620, 1150, 626, 2578, 300]
                                    ], dtype=np.int32)
    #front
    # static_guide_points = np.array([
    #                                 [512, 450, 753 , 455, 2578, 3000],
    #                                 [461, 473, 802 , 481, 2578, 2000],
    #                                 [353, 523, 906 , 536, 2578, 1000],
    #                                 [256, 569, 1001, 586, 2578, 500],
    #                                 [209, 591, 1047, 611, 2578, 300],
    #                                 ], dtype=np.int32)
    ## Lux SA
    # static_guide_points = np.array([[525, 363, 815, 368, 2578, 3000 ],
    #                                 [470, 400, 863, 398, 2578 ,2000],
    #                                 [357, 474, 976, 474, 2578 ,1000],
    #                                 [243, 548, 1100, 555,2578, 500 ],
    #                                 [164, 599, 1166, 597,2578, 300 ]
    #                             ], dtype=np.int32)
    if args.demo == "image":
        if os.path.isdir(args.path):
            files = get_image_list(args.path)
        else:
            files = [args.path]
        files.sort()
        for image_name in files:
            meta, res = predictor.inference(image_name)
            result_image = predictor.visualize(res[0], meta, cfg.class_names, 0.4)
            if args.save_result:
                save_folder = os.path.join(
                    './', time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
                )
                mkdir(local_rank, save_folder)
                save_file_name = os.path.join(save_folder, os.path.basename(image_name))
                cv2.imwrite("test.jpg", result_image)
            ch = cv2.waitKey(0)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break

    elif args.demo == "image_folder":
        image_list = sorted(glob(os.path.join(args.path,"*.jpeg")))
        # image_list = sorted(glob(os.path.join(args.path,"image_0*")))
        # image_list = sorted(glob(os.path.join(args.path,"*.jpg")))
        print(f"Total Frame: {len(image_list)}")
        os.makedirs("./dump", exist_ok=True)
        save_path = f"demo_images_track_9.mp4"
        vid_writer = imageio.get_writer(save_path, format='mp4', mode='I', fps=10)
        time_accum = 0
        frame_count = 0
        timer = Timer()

        mot_tracker = None
        for imp in tqdm(image_list):
            t0 = time.perf_counter()
            timer.tic()
            frame = cv2.imread(imp)
            meta, res = predictor.inference(frame)
            img_h, img_w = meta['img_info']['height'][0], meta['img_info']['width'][0]
            if res[0] is not None:
                if args.track:
                    if mot_tracker is None:
                        mot_tracker = SORT(
                            max_age=args.max_age,
                            min_hits=args.min_hits,
                            iou_threshold=args.iou_threshold,
                            shape=[frame.shape[0], frame.shape[1]],
                            camera_fps=args.fps,
                            ttc_type=args.ttc_type,
                            focal_length=args.focal_length,
                            camera_height=args.camera_height
                        )
                    rects = []
                    for label, bboxes in res[0].items():
                        class_id = int(label)
                        if len(bboxes) > 0:
                            mask = np.array(bboxes)[:,-1] > 0.3
                            rects.append(np.array(np.array(bboxes)[mask]))
                    if len(rects) == 0:
                        rects = np.empty((0, 5))
                    rects = np.concatenate(rects)
                    trackers = mot_tracker.update(rects)
                    timer.toc()
                    contour = np.array(static_guide_points[:5,:4]).reshape(10,2)
                    for trk in trackers:
                        bbox = trk.get_state()[0]
                        bbox_contour = rect2poly(bbox.astype(int))
                        iou = polyiou_overlap(Polygon(contour), Polygon(bbox_contour))
                        print(iou)
                        if iou > 0:
                            visualize(im, trk, args, contour)
                    result_frame = im
                else:
                    im = drawGuideLines(frame, static_guide_points)
                    rects = []
                    for label, bboxes in res[0].items():
                        class_id = int(label)
                        if len(bboxes) > 0:
                            mask = np.array(bboxes)[:,-1] > 0.4
                            rects.append(np.array(np.array(bboxes)[mask]))
                    if len(rects) == 0:
                        rects = np.empty((0, 5))

                    rects = wbf(np.concatenate(rects), img_w, img_h)                    
                    valid_rects = []
                    contour = np.array(static_guide_points[0:5,:4]).reshape(10,2)
                    contour_update = [tuple(contour[0]), tuple(contour[2]), tuple(contour[8]), (0,img_h),
                                      (img_w, img_h),tuple(contour[9]), tuple(contour[3]), tuple(contour[1]), 
                                     ]
                    for bbox in rects:
                        # bbox_contour = rect2poly(bbox.astype(int))
                        # iou = polyiou_overlap(Polygon(contour), Polygon(bbox_contour))
                        # if iou > 0:
                        ## add point to check
                        points_to_check = points2check(bbox)
                        check_nearby = is_inside_polygon(contour_update, points_to_check)
                        if np.array(check_nearby).mean() > 0.5:
                            valid_rects.append(bbox)
                    if len(valid_rects) > 0:
                        im = plot_det(im, valid_rects, color=(0,255,0))
                        # distance = [np.linalg.norm(np.array(box[2], box[3]) - np.array((box[2],img_h))) for box in valid_rects]
                        # distances = [distance.euclidean(np.array(box[2], box[3]),np.array((box[2],img_h))) for box in valid_rects]
                        distances = [np.sqrt((box[3] - img_h)**2) for box in valid_rects]
                        idx = np.argmin(np.array(distances))
                        threat = [valid_rects[idx]] 
                    else:
                        threat = valid_rects
                    im = plot_det(im, threat, color=(0,0,255))

                    result_frame = im

            time_accum = time_accum + (time.perf_counter()-t0)
            frame_count+=1
            online_im_rgb = result_frame #cv2.cvtColor(online_im, cv2.COLOR_BGR2RGB)

            out_im = online_im_rgb.copy()
            # alpha = 0.75
            # mask = shape.astype(bool)
            # out_im[mask] = cv2.addWeighted(online_im_rgb, alpha, shape, 1 - alpha, 0)[mask]
        
            cv2.imwrite(f"dump/test_{frame_count}.png",out_im)
            vid_writer.append_data(cv2.cvtColor(out_im,cv2.COLOR_BGR2RGB))

        print(time_accum)
        print("fps: ", frame_count/time_accum)
        vid_writer.close()

if __name__ == "__main__":
    main()
