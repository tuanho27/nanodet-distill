# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import polygonize
from .base import BaseDataset


class CocoSegDataset(BaseDataset):
    def get_data_info(self, ann_path):
        """
        Load basic information of dataset such as image path, label and so on.
        :param ann_path: coco json file path
        :return: image info:
        [{'license': 2,
          'file_name': '000000000139.jpg',
          'coco_url': 'http://images.cocodataset.org/val2017/000000000139.jpg',
          'height': 426,
          'width': 640,
          'date_captured': '2013-11-21 01:34:01',
          'flickr_url':
              'http://farm9.staticflickr.com/8035/8024364858_9c41dc1666_z.jpg',
          'id': 139},
         ...
        ]
        """
        self.coco_api = COCO(ann_path)
        self.cat_ids = sorted(self.coco_api.getCatIds())
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cats = self.coco_api.loadCats(self.cat_ids)
        self.class_names = [cat["name"] for cat in self.cats]
        self.img_ids = sorted(self.coco_api.imgs.keys())
        img_info = self.coco_api.loadImgs(self.img_ids)
        return img_info

    def get_per_img_info(self, idx):
        img_info = self.data_info[idx]
        file_name = img_info["file_name"]
        height = img_info["height"]
        width = img_info["width"]
        id = img_info["id"]
        if not isinstance(id, int):
            raise TypeError("Image id must be int.")
        info = {"file_name": file_name, "height": height, "width": width, "id": id}
        return info

    def get_extreme_points(self, pts):
        l, t = min(pts[:, 0]), min(pts[:, 1])
        r, b = max(pts[:, 0]), max(pts[:, 1])
        # 3 degrees
        thresh = 0.02
        print(f"xxx{l}")
        print(f"yyy{r}")
        w = r - l + 1
        h = b - t + 1

        t_idx = np.argmin(pts[:, 1])
        t_idxs = [t_idx]
        tmp = (t_idx + 1) % pts.shape[0]
        while tmp != t_idx and pts[tmp, 1] - pts[t_idx, 1] <= thresh * h:
            t_idxs.append(tmp)
            tmp = (tmp + 1) % pts.shape[0]
        tmp = (t_idx - 1) % pts.shape[0]
        while tmp != t_idx and pts[tmp, 1] - pts[t_idx, 1] <= thresh * h:
            t_idxs.append(tmp)
            tmp = (tmp - 1) % pts.shape[0]
        tt = [(max(pts[t_idxs, 0]) + min(pts[t_idxs, 0])) / 2, t]

        b_idx = np.argmax(pts[:, 1])
        b_idxs = [b_idx]
        tmp = (b_idx + 1) % pts.shape[0]
        while tmp != b_idx and pts[b_idx, 1] - pts[tmp, 1] <= thresh * h:
            b_idxs.append(tmp)
            tmp = (tmp + 1) % pts.shape[0]
        tmp = (b_idx - 1) % pts.shape[0]
        while tmp != b_idx and pts[b_idx, 1] - pts[tmp, 1] <= thresh * h:
            b_idxs.append(tmp)
            tmp = (tmp - 1) % pts.shape[0]
        bb = [(max(pts[b_idxs, 0]) + min(pts[b_idxs, 0])) / 2, b]

        l_idx = np.argmin(pts[:, 0])
        l_idxs = [l_idx]
        tmp = (l_idx + 1) % pts.shape[0]
        while tmp != l_idx and pts[tmp, 0] - pts[l_idx, 0] <= thresh * w:
            l_idxs.append(tmp)
            tmp = (tmp + 1) % pts.shape[0]
        tmp = (l_idx - 1) % pts.shape[0]
        while tmp != l_idx and pts[tmp, 0] - pts[l_idx, 0] <= thresh * w:
            l_idxs.append(tmp)
            tmp = (tmp - 1) % pts.shape[0]
        ll = [l, (max(pts[l_idxs, 1]) + min(pts[l_idxs, 1])) / 2]

        r_idx = np.argmax(pts[:, 0])
        r_idxs = [r_idx]
        tmp = (r_idx + 1) % pts.shape[0]
        while tmp != r_idx and pts[r_idx, 0] - pts[tmp, 0] <= thresh * w:
            r_idxs.append(tmp)
            tmp = (tmp + 1) % pts.shape[0]
        tmp = (r_idx - 1) % pts.shape[0]
        while tmp != r_idx and pts[r_idx, 0] - pts[tmp, 0] <= thresh * w:
            r_idxs.append(tmp)
            tmp = (tmp - 1) % pts.shape[0]
        rr = [r, (max(pts[r_idxs, 1]) + min(pts[r_idxs, 1])) / 2]

        return np.array([tt, ll, bb, rr])


    def filter_tiny_polys(self, polys):
        polys_ = []
        for poly in polys:
            x_min, y_min = np.min(poly[:, 0]), np.min(poly[:, 1])
            x_max, y_max = np.max(poly[:, 0]), np.max(poly[:, 1])
            if x_max - x_min >= 1 and y_max - y_min >= 1:
                polys_.append(poly)
        return [poly for poly in polys_ if Polygon(poly).area > 5]

    def get_cw_polys(self, polys):
        return [poly[::-1] if Polygon(poly).exterior.is_ccw else poly for poly in polys]

    def get_valid_polys(self, instance_polys, inp_out_hw):
        # output_h, output_w = inp_out_hw[2:]
        output_h, output_w = inp_out_hw['height'], inp_out_hw['width']
        instance_polys_ = []
        for instance in instance_polys:
            instance = [poly for poly in instance if len(poly) >= 4]
            for poly in instance:
                poly[:, 0] = np.clip(poly[:, 0], 0, output_w - 1)
                poly[:, 1] = np.clip(poly[:, 1], 0, output_h - 1)
            polys = self.filter_tiny_polys(instance)
            polys = self.get_cw_polys(polys)
            polys = [poly[np.sort(np.unique(poly, axis=0, return_index=True)[1])] for poly in polys]
            polys = [poly for poly in polys if len(poly) >= 4]
            instance_polys_.append(polys)
        return instance_polys_

    def get_img_annotation(self, idx, img_info):
        """
        load per image annotation
        :param idx: index in dataloader
        :return: annotation dict
        """
        print("IDX",idx)
        img_id = self.img_ids[idx]
        ann_ids = self.coco_api.getAnnIds([img_id])
        anns = self.coco_api.loadAnns(ann_ids)
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks = []
        gt_mask_polys = []
        gt_poly_lens = []

        if self.use_keypoint:
            gt_keypoints = []
        for ann in anns:
            if ann.get("ignore", False):
                continue
            x1, y1, w, h = ann["bbox"]
            if ann["area"] <= 0 or w < 1 or h < 1:
                continue
            if ann["category_id"] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get("iscrowd", False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann["category_id"]])
                gt_masks.append(self.coco_api.annToMask(ann))
                if self.use_keypoint:
                    gt_keypoints.append(ann["keypoints"])
                ##    
                mask_polys = [
                    np.array(p).reshape(-1, 2) for p in ann['segmentation'] if len(p) >= 6
                ]  # valid polygons have >= 3 points (6 coordinates)
                print(mask_polys)
                poly_lens = [len(p) for p in mask_polys]
                gt_mask_polys.append(mask_polys)
                gt_poly_lens.extend(poly_lens)
                    
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
        annotation = dict(
            bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore
        )
        annotation["masks"] = gt_masks
        if self.use_keypoint:
            if gt_keypoints:
                annotation["keypoints"] = np.array(gt_keypoints, dtype=np.float32)
            else:
                annotation["keypoints"] = np.zeros((0, 51), dtype=np.float32)
        ##
        instance_polys = self.get_valid_polys(gt_mask_polys, img_info)

        points = [self.get_extreme_points(poly) for poly in instance_polys]

        annotation["ext_cts"] = points

        return annotation

    def get_train_data(self, idx):
        """
        Load image and annotation
        :param idx:
        :return: meta-data (a dict containing image, annotation and other information)
        """
        img_info = self.get_per_img_info(idx)
        file_name = img_info["file_name"]
        image_path = os.path.join(self.img_path, file_name)
        img = cv2.imread(image_path)
        if img is None:
            print("image {} read failed.".format(image_path))
            raise FileNotFoundError("Cant load image! Please check image path!")
        ann = self.get_img_annotation(idx, img_info)
        meta = dict(
            img=img, img_info=img_info, gt_bboxes=ann["bboxes"], gt_labels=ann["labels"]
        )
        meta["gt_masks"] = ann["masks"]
        meta["gt_ext_contours"] = ann["ext_cts"]
        if self.use_keypoint:
            meta["gt_keypoints"] = ann["keypoints"]

        input_size = self.input_size
        if self.multi_scale:
            print("Multi Scale @@@@")
            input_size = self.get_random_size(self.multi_scale, input_size)

        meta = self.pipeline(self, meta, input_size)

        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1))
        return meta

    def get_val_data(self, idx):
        """
        Currently no difference from get_train_data.
        Not support TTA(testing time augmentation) yet.
        :param idx:
        :return:
        """
        # TODO: support TTA
        return self.get_train_data(idx)
