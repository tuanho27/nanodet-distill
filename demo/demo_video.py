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

from nanodet.collision.sort_tracker.sorttracker import SORT, KalmanBoxTracker
from nanodet.collision.sort_tracker.utils import *
from nanodet.collision.timer import Timer
from nanodet.util.helpers import *

from nanodet.util.visualize import plot_tracking
from nanodet.collision.byte_tracker.byte_tracker import BYTETracker
from nanodet.collision.topview_handler import TopView360Handler

image_ext = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
video_ext = ["mp4", "mov", "avi", "mkv"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("--config", help="model config file path")
    parser.add_argument(
        "--calib", default="calib/", help="path calib folder data"
    )
    parser.add_argument("--model", help="model file path")
    parser.add_argument("--path", default="./demo", help="path to images or video")
    parser.add_argument("--track", action='store_true', help="Add tracking by default")
    parser.add_argument("--min_box_area", default=100, type=float, help="min box area to eliminate noise")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )

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
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )                        

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
    elif args.demo == "video" or args.demo == "webcam":
        cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        fps = cap.get(cv2.CAP_PROP_FPS)
        save_path = f"demo_images_track_3.mp4"
        vid_writer = imageio.get_writer(save_path, format='mp4', mode='I', fps=10)
        timer = Timer()
        frame_count = 0
        time_accum = 0
        mot_tracker = None
        
        for _ in tqdm(generator()):
            ret_val, frame = cap.read()
            if ret_val:
                t0 = time.perf_counter()
                timer.tic()
                meta, res = predictor.inference(frame)
                if res[0] is not None:
                    ######## For collision warning #########
                    cw_track_region = cw_track_region = get_track_region("side")
                    shape =  np.zeros_like(frame, np.uint8)
                    cv2.rectangle(
                        shape,
                        (cw_track_region[0], cw_track_region[1]),
                        (cw_track_region[2], cw_track_region[3]),
                        (20,150,20),
                        -1
                    )
                ########################################
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

                    try:
                        rects = np.concatenate(rects)
                        ious_overlap = ious(rects[:,:4],cw_track_region.reshape(1,4))
                        valid_index = ious_overlap > 0
                        rects = rects[valid_index[:,0]]
                        trackers = mot_tracker.update(rects)
                        timer.toc()
                        contour = np.array([args.guidelines], dtype=np.int32)
                        im = drawGuideLines(frame, contour)
                        for trk in trackers:
                            visualize(im, trk, args, contour)
                        result_frame = im

                    except:
                        result_frame = frame
                time_accum = time_accum + (time.perf_counter()-t0)
                frame_count+=1
                online_im_rgb = result_frame #cv2.cvtColor(online_im, cv2.COLOR_BGR2RGB)

                out_im = online_im_rgb.copy()
                alpha = 0.75
                mask = shape.astype(bool)
                out_im[mask] = cv2.addWeighted(online_im_rgb, alpha, shape, 1 - alpha, 0)[mask]

                cv2.imwrite(f"dump/test_{frame_count}.png",out_im)
                vid_writer.append_data(cv2.cvtColor(out_im,cv2.COLOR_BGR2RGB))

                if frame_count==500:
                    break
                
                # if args.save_result:
                #     vid_writer.write(result_frame)
                # ch = cv2.waitKey(1)
                # if ch == 27 or ch == ord("q") or ch == ord("Q"):
                #     break
            else:
                break
        print(time_accum)
        print("fps: ", frame_count/time_accum)
        vid_writer.close()

if __name__ == "__main__":
    main()
