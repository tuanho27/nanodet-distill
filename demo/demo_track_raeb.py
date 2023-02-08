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
from natsort import natsorted # pip install natsort

from collections import OrderedDict
from shapely.geometry import Polygon
from shapely.validation import make_valid
from scipy.spatial import distance
# from ensemble_boxes import weighted_boxes_fusion
from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from nanodet.data.transform import Pipeline
from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight
from nanodet.util.path import mkdir
from nanodet.util.helpers import *

from nanodet.collision.sort_tracker.sorttracker import SORT, KalmanBoxTracker
from nanodet.collision.sort_tracker.utils import *
from nanodet.collision.timer import Timer

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
    parser.add_argument("--model", help="model file path")
    parser.add_argument(
        "--calib", default="calib_high/", help="path calib folder data"
    )
    parser.add_argument("--path", default="./demo", help="path to images or video")
    parser.add_argument("--reID_model_path", default="./yolov7-deepsort-tracking/deep_sort/model_weights/mars-small128.pb", help="path to deepsort models")
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
    parser.add_argument("--conf", default=0.15, type=float, help="test conf")

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
    parser.add_argument("--track_buffer", type=int, default=200, help="the frames for keep lost tracks")
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
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")

    args = parser.parse_args()
    return args


class Predictor(object):
    def __init__(self, cfg, model_path, logger, device="cuda:0", distill=False):
        self.cfg = cfg
        self.device = device
        self.distill = distill
        model = build_model(cfg.model)
            
        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        if self.distill:
            new_state_dict = OrderedDict()
            for k, v in ckpt["state_dict"].items():
                if not k.startswith('teacher_model') and "conv1x1" not in k:
                    new_state_dict[k] = v
            new_checkpoint = dict(state_dict=new_state_dict)
            load_model_weight(model, new_checkpoint, logger)
        else:
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

def get_new_output_format(inputs, thresh):
    outputs = []
    for k,v in inputs.items():
        if len(v) > 0:
            mask = np.array(v)[:,-1] > thresh
            for data in np.array(np.array(v)[mask]): 
                outputs.append(data.tolist() + [k])
    return torch.tensor(outputs)

def main():
    args = parse_args()
    local_rank = 0
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    load_config(cfg, args.config)
    logger = Logger(local_rank, use_tensorboard=False)
    predictor = Predictor(cfg, args.model, logger, device="cuda:0", distill=True)
    if args.fp16:
        predictor = predictor.half()  # to FP16
        
    logger.log('Press "Esc", "q" or "Q" to exit.')
    current_time = time.localtime()
    static_guide_points = read_guideline(['./calib/e34/object_detection_area_rear_cam.txt'])[0]

    # creat output folder
    args.experiment_name = args.config.split("/")[-1].split(".yml")[0]

    output_dir = os.path.join("./outputs/", args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    if args.save_result:
        vis_folder = os.path.join(output_dir, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)
    
    # read calib
    with open(f'{args.calib}/cameraData.json', 'r') as f:
        calib_data = json.load(f)
    calib_data = calib_data['Items']

    tracker = BYTETracker(args, frame_rate=args.fps)
    topview_size = (180, 160)
    vis_scale = 2
    topview_image_rgb = np.zeros((topview_size[0]*vis_scale, topview_size[1]*vis_scale, 3))
    topview_image = np.zeros((topview_size[0]*vis_scale, topview_size[1]*vis_scale, 3))
    bev_mapping = TopView360Handler(calib_data, num_channels=3, dX=0.1, dY = 0.1, grid_rows=topview_size[0], grid_cols=topview_size[1], borderValue=0)
    vid_name = args.path.split("/")[-1]
    vid_writer = imageio.get_writer(f"{vis_folder}/../{vid_name}.mp4", format='mp4', mode='I', fps=20)

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
                cv2.imwrite(f"{save_file_name}", result_image)
            ch = cv2.waitKey(0)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break

    elif args.demo == "image_folder":
        image_list = natsorted(glob(os.path.join(args.path,"*.png")))
        print(f"Total Frame: {len(image_list)}")
        time_accum = 0
        frame_count = 0
        results = []
        timer = Timer()
        frame_id = 0
        for imp in tqdm(image_list):
            name = imp.split("/")[-1]
            t0 = time.perf_counter()
            timer.tic()
            frame = cv2.imread(imp)
            meta, res = predictor.inference(frame)
            img_h, img_w = meta['img_info']['height'][0], meta['img_info']['width'][0]
            outputs = get_new_output_format(res[0], args.conf)
            if len(outputs) > 0:
                online_targets = tracker.update(outputs, meta['raw_img'][0], [img_h, img_w], tuple([img_h, img_w])) #nanodet already re-scale the output.
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        # save results
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                timer.toc()
                online_im = plot_tracking(
                    meta['raw_img'][0], online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time
                )
                topview_image = bev_mapping.process(meta['raw_img'][0], online_im, online_tlwhs, online_ids)
                # topview_image = bev_mapping.process(online_im, online_tlwhs, online_ids)
                topview_image, online_im = cv2.resize(topview_image, None, fx = vis_scale, fy = vis_scale, interpolation = cv2.INTER_CUBIC)

                ######################
                static_guide_points =  np.concatenate((static_guide_points[0:5,:4],np.array([[0,img_h, img_w, img_h]])))
                online_im = drawGuideLines(online_im, static_guide_points)

            else:
                timer.toc()
                online_im = meta['raw_img'][0]

            # result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
            if args.save_result:
                timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
                save_folder = os.path.join(vis_folder, timestamp)
                os.makedirs(save_folder, exist_ok=True)
                cv2.imwrite(os.path.join(save_folder, os.path.basename(imp)), online_im)
                # vid_writer.append_data(cv2.cvtColor(online_im, cv2.COLOR_BGR2RGB))
                vis_image = np.zeros((800,1800,3),dtype=np.uint8)
                vis_image[:,:1280,:] = online_im
                vis_image[220:580,1400:1720,:] = topview_image
                vid_writer.append_data(vis_image)
                cv2.imwrite("test.jpg", cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))

            frame_id
            time_accum = time_accum + (time.perf_counter()-t0)
            frame_count+=1
    else:
        print("Demo with video input: {args.path}")
        reader = imageio.get_reader(f"{args.path}")
        print(f"Total Frame: {len(reader)}")
        time_accum = 0
        frame_count = 0
        results = []
        timer = Timer()
        frame_id = 0
        for frame_number, im in enumerate(reader):
            print(f"Frame: {frame_number}")
            frame_id = frame_number
            t0 = time.perf_counter()
            timer.tic()
            imtl = im[:400,:640,:]
            imtr = im[:400,640:,:]
            imbl = im[400:,:640,:]
            imbr = im[400:,640:,:]
            frame = imtl
            meta, res = predictor.inference(frame)
            img_h, img_w = meta['img_info']['height'][0], meta['img_info']['width'][0]
            outputs = get_new_output_format(res[0], args.conf)
            if len(outputs) > 0:
                online_targets = tracker.update(outputs, meta['raw_img'][0], [img_h, img_w], tuple([img_h, img_w])) #nanodet already re-scale the output.
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        # save results
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                timer.toc()
                online_im = plot_tracking(
                    meta['raw_img'][0], online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time
                )
                topview_image,online_im = bev_mapping.process(meta['raw_img'][0], online_im, online_tlwhs, online_ids, frame_id)

                topview_image = cv2.resize(topview_image, None, fx = vis_scale, fy = vis_scale, interpolation = cv2.INTER_CUBIC)
                data_rgb = {
                            "front": cv2.resize(imbr, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC),
                            "left": cv2.resize(imbl, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC),
                            "rear": cv2.resize(imtl, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC),
                            "right":cv2.resize(imtr, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)
                }
                topview_image_rgb = bev_mapping.handle(data_rgb)
                topview_image_rgb = cv2.resize(topview_image_rgb, None, fx = vis_scale, fy = vis_scale, interpolation = cv2.INTER_CUBIC)


                ######################
                static_guide_points =  np.concatenate((static_guide_points[0:5,:4],np.array([[0,img_h, img_w, img_h]])))
                online_im = drawGuideLines(online_im, (static_guide_points/2).astype(int))
            else:
                timer.toc()
                online_im = meta['raw_img'][0]

            # result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
            if args.save_result:
                timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
                save_folder = os.path.join(vis_folder, timestamp)
                os.makedirs(save_folder, exist_ok=True)
                cv2.imwrite(f"{save_folder}/{frame_number:04}.jpg", online_im)
                # vid_writer.append_data(cv2.cvtColor(online_im, cv2.COLOR_BGR2RGB))
                vis_image = np.zeros((440,1650,3),dtype=np.uint8)
                vis_image[20:420,:640,:] = online_im
                vis_image[20:180, 640:896] = cv2.resize(im, None,  fx = 1/5, fy = 1/5, interpolation = cv2.INTER_CUBIC)

                vis_image[40:400,930:1250,:] = topview_image_rgb
                vis_image[40:400,1280:1600,:] = topview_image
                vid_writer.append_data(vis_image)
                cv2.imwrite("test.jpg", cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))

            frame_id
            time_accum = time_accum + (time.perf_counter()-t0)
            frame_count+=1


    print("fps: ", frame_count/time_accum)
    vid_writer.close()

if __name__ == "__main__":
    main()
