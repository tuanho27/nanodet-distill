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

image_ext = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
video_ext = ["mp4", "mov", "avi", "mkv"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="model config file path")
    parser.add_argument("--model", help="model file path")
    parser.add_argument("--path", default="/data/parking/dataset/cams_1666345462836/", help="path to images or video")
    parser.add_argument("-iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    parser.add_argument("-debug", action='store_true', help="Minimum IOU for match.")
    parser.add_argument("--save_result", action="store_true", help="whether to save the inference result of image/video")
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


def main():
    args = parse_args()
    local_rank = 0
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    load_config(cfg, args.config)
    logger = Logger(local_rank, use_tensorboard=False)
    predictor = Predictor(cfg, args.model, logger, device="cuda:0", distill=True)
    # image_fronts = sorted(glob(args.path + "/im_front*"))
    # image_lefts = sorted(glob(args.path + "/im_left*"))
    # image_rears = sorted(glob(args.path + "/im_rear*"))
    # image_rights = sorted(glob(args.path + "/im_right*"))
    # for i in range(len(image_fronts)):
    save_path = f"demo_od_parking_perpendicular_.mp4"
    vid_writer = imageio.get_writer(save_path, format='mp4', mode='I', fps=10)

    # perpendicular
    # reader = imageio.get_reader('/data/parking/dataset/cams_1666774864737.mp4')
    # reader1 =  imageio.get_reader('/data/parking/dataset/topview_1666774864737_seg.mp4')

    # parallel
    reader = imageio.get_reader('/data/parking/dataset/fisheyes/cams_1666774995681.mp4')
    reader1 =  imageio.get_reader('/data/parking/dataset/topview/topview_1666774995681.mp4')
    for frame_number, (im, im_top) in enumerate(zip(reader, reader1)):
        
        if frame_number > 300:
            print(frame_number)

            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            out_im  = np.zeros((1056,2100,3))
            imtl = im[:400,:640,:]
            imtr = im[:400,640:,:]
            imbl = im[400:,:640,:]
            imbr = im[400:,640:,:]
            result_images = []
            for img in [imtl,imtr,imbl,imbr]:
                meta, res = predictor.inference(img)
                result_images.append(predictor.visualize(res[0], meta, cfg.class_names, 0.65))
            out_im[100:500,:640,:] =  cv2.cvtColor(result_images[0],cv2.COLOR_BGR2RGB)
            out_im[100:500,640:1280,:] = cv2.cvtColor(result_images[1],cv2.COLOR_BGR2RGB)
            out_im[500:900,:640,:] =  cv2.cvtColor(result_images[2],cv2.COLOR_BGR2RGB)
            out_im[500:900,640:1280,:] = cv2.cvtColor(result_images[3],cv2.COLOR_BGR2RGB)
            out_im[:,1300:,:] = im_top
            vid_writer.append_data(out_im)
            # vid_writer.append_data(cv2.cvtColor(out_im,cv2.COLOR_BGR2RGB))

            # cv2.imwrite(f"im_rear_{frame_number}.jpg",imtl)
            # cv2.imwrite(f"im_left_{frame_number}.jpg",imtr)
            # cv2.imwrite(f"im_right_{frame_number}.jpg",imbl)
            # cv2.imwrite(f"im_front_{frame_number}.jpg",imbr)
            cv2.imwrite(f"im_topview_{frame_number}.jpg", cv2.cvtColor(im_top,cv2.COLOR_BGR2RGB))
            # subprocess.run(["./OpenGL", 
            #                     image_fronts[i],
            #                     image_lefts[i],
            #                     image_rears[i],
            #                     image_rights[i]], 
            #                     stdout = subprocess.PIPE, universal_newlines = True).stdout

    vid_writer.close()

if __name__ == "__main__":
    main()
