from ipaddress import ip_address
from statistics import variance
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import json
import sys
import math
import os
from nanodet.util.visualize import _COLORS, get_color
from bisect import bisect_left
from .byte_tracker.byte_tracker import STrack
from .byte_tracker.kalman_filter import KalmanFilter

calib_folder = "calib_high/" 

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def take_closest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    min(myList, key=lambda x:abs(x-myNumber))
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after
    else:
        return before

class TopView360Handler:
    CAM_LEFT = 0
    CAM_FRONT = 1
    CAM_REAR = 2
    CAM_RIGHT = 3
    def __init__(self, calib_data, num_channels=3, dX=0.015, dY = 0.015, grid_rows=1000, grid_cols=1000, borderValue=0) -> None:
        self.fisheye_shape = (1280, 800)
        self.dX = dX
        self.dY = dY
        K_G = np.zeros((3, 3))
        K_G[0, 0] = 1 / dX
        K_G[1, 1] = -1 / dY
        K_G[0, 2] = grid_cols / 2
        K_G[1, 2] = grid_rows / 2
        K_G[2, 2] = 1.0
        self.K_G_inv = np.linalg.inv(K_G)

        self.uv_maps = []
        self.fisheye_to_undistorted_mappings = []
        self.world_idxs = []
        
        for i in range(4):
            uv_map, fisheye_to_undistorted_mapping, world_idx = self.precompute_uvmap(grid_rows, grid_cols, calib_data[i])
            self.uv_maps.append(uv_map)
            self.fisheye_to_undistorted_mappings.append(fisheye_to_undistorted_mapping)
            self.world_idxs.append(world_idx)
        # self.uv_map = np.loadtxt("calib/rearVUV.txt")
        # half_car_w_in_px= int(2.9 / dX)
        # half_car_h_in_px = int(1.3 / dY)
        half_car_w_in_px= int(2.3 / dX)
        half_car_h_in_px = int(1.0 / dY)

        alphas = [cv2.imread(f'{calib_folder}/alpha_{i}.png', cv2.IMREAD_GRAYSCALE) for i in range(4)]
        bev_alphas = [cv2.remap(im, uv_map, None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,borderValue=0) for im, uv_map in zip(alphas, self.uv_maps)]
        bev_alphas[self.CAM_RIGHT][:grid_rows // 2 + half_car_h_in_px] = 0
        bev_alphas[self.CAM_LEFT][grid_rows // 2 - half_car_h_in_px:] = 0
        bev_alphas[self.CAM_REAR][:,grid_cols // 2 - half_car_w_in_px:] = 0
        bev_alphas[self.CAM_FRONT][:,:grid_cols // 2 + half_car_w_in_px] = 0
        self.bev_alphas = [(alpha != 0) for alpha in bev_alphas]
        if num_channels != 1:
            self.topview_size = (grid_rows, grid_cols, num_channels)
        else:
            self.topview_size = (grid_rows, grid_cols)
        self.borderValue = borderValue
        self.bev_images=[]
        self.topview = np.zeros(self.topview_size, dtype=np.uint8) 
        # add car shape in the middle
        self.topview[self.bev_alphas[self.CAM_FRONT]] = np.ones_like(self.topview[self.bev_alphas[self.CAM_FRONT]])*255
        self.topview[self.bev_alphas[self.CAM_LEFT]]  = np.ones_like(self.topview[self.bev_alphas[self.CAM_LEFT]])*255
        self.topview[self.bev_alphas[self.CAM_RIGHT]] = np.ones_like(self.topview[self.bev_alphas[self.CAM_RIGHT]])*255
        self.topview[self.bev_alphas[self.CAM_REAR]]  = np.ones_like(self.topview[self.bev_alphas[self.CAM_REAR]])*255
        self.mappixels = self.extrapolate_uvmap() 

        #camera instrinsics & extrinsics
        self.matD, self.matK, self.matR, self.vecT, self.matR_inv = self.ReadCameraData(f'./{calib_folder}/cameraData.json')
        self.track_data = {}
        self.track_objs = {}

    #read calib data from file
    def ReadCameraData(self, path):
        my_file         =  open(path, 'r')
        exstrinsic_data = json.load(my_file)
        camera_matD     = np.zeros((4,4))
        camera_matK     = np.zeros((4,4))
        camera_matR_    = np.zeros((4,9))
        camera_vecT     = np.zeros((4,3))
        camera_matR     = np.zeros((4,3,3))
        camera_matRinv  = np.zeros((4,3,3))
        for i in range(4):
            camera_matD[i]  = exstrinsic_data['Items'][i]['matrixD']
            camera_matK[i]  = exstrinsic_data['Items'][i]['matrixK']
            camera_matR_[i] = exstrinsic_data['Items'][i]['matrixR']
            camera_vecT[i]  = exstrinsic_data['Items'][i]['vectT']
            camera_matR[i]  = np.reshape(camera_matR_[i], (3, 3))
            camera_matRinv[i] = np.linalg.inv(camera_matR[i])
        return camera_matD, camera_matK, camera_matR, camera_vecT, camera_matRinv

    #distort function
    def distort(self,theta, mat_D):
        thetaSq = theta*theta
        cdist = theta * (1.0 + mat_D[0] * thetaSq + 
                                    mat_D[1] * thetaSq * thetaSq +
                                    mat_D[2] * thetaSq * thetaSq * thetaSq +
                                    mat_D[3] * thetaSq * thetaSq * thetaSq * thetaSq)
        return cdist

    def pixcel2ray(self, px, py, camPos):
        #pixcel to sensor coordinate
        xd = float(px - self.matK[camPos][1])/self.matK[camPos][0]
        yd = float(py - self.matK[camPos][3])/self.matK[camPos][2]
        cdist = math.sqrt(xd*xd + yd*yd)
        #theta solver
        y = cdist
        cont = 1
        a = 0.0
        b = 3.14/2
        fa = self.distort(a, self.matD[camPos]) - y
        fb = self.distort(b, self.matD[camPos]) - y
        while cont > 0:
            x = a - (b-a)*fa/(fb-fa)
            fx = self.distort(x, self.matD[camPos]) - y
            if ( abs(fx) < 0.000001):
                cont = 0
            if (fx*fa < 0):
                b = x
            else:
                a = x
        theta = x
        len3D = 1/math.sin(theta)
        xc = xd/cdist
        yc = yd/cdist
        zc = math.sqrt(len3D*len3D - 1)
        #print(xc, yc, zc)

        #change camera position to global coordinate
        x_0 = self.matR_inv[camPos][0][0]*(-self.vecT[camPos][0]) + self.matR_inv[camPos][0][1]*(-self.vecT[camPos][1]) + self.matR_inv[camPos][0][2]*(-self.vecT[camPos][2])
        y_0 = self.matR_inv[camPos][1][0]*(-self.vecT[camPos][0]) + self.matR_inv[camPos][1][1]*(-self.vecT[camPos][1]) + self.matR_inv[camPos][1][2]*(-self.vecT[camPos][2])
        z_0 = self.matR_inv[camPos][2][0]*(-self.vecT[camPos][0]) + self.matR_inv[camPos][2][1]*(-self.vecT[camPos][1]) + self.matR_inv[camPos][2][2]*(-self.vecT[camPos][2])

        #change second point to global coordinate
        x_1 = self.matR_inv[camPos][0][0]*(xc-self.vecT[camPos][0]) + self.matR_inv[camPos][0][1]*(yc-self.vecT[camPos][1]) + self.matR_inv[camPos][0][2]*(zc-self.vecT[camPos][2])
        y_1 = self.matR_inv[camPos][1][0]*(xc-self.vecT[camPos][0]) + self.matR_inv[camPos][1][1]*(yc-self.vecT[camPos][1]) + self.matR_inv[camPos][1][2]*(zc-self.vecT[camPos][2])
        z_1 = self.matR_inv[camPos][2][0]*(xc-self.vecT[camPos][0]) + self.matR_inv[camPos][2][1]*(yc-self.vecT[camPos][1]) + self.matR_inv[camPos][2][2]*(zc-self.vecT[camPos][2])
        #print(x_1,y_1,z_1)
        return x_0, y_0, z_0, x_1, y_1, z_1

    def intersectionZ(self,x_0, y_0, z_0, x_1, y_1, z_1, z_global):
        #find intersection point
        #z_global = 0
        x_global = x_0 + (x_1 - x_0)*(z_global - z_0)/(z_1 - z_0)
        y_global = y_0 + (y_1 - y_0)*(z_global - z_0)/(z_1 - z_0)
        #print(x_global, y_global, z_global)
        return x_global, y_global

    def intersectionX(self,x_0, y_0, z_0, x_1, y_1, z_1, x_global):
        #find intersection point
        #z_global = 0
        z_global = z_0 + (z_1 - z_0)*(x_global - x_0)/(x_1 - x_0)
        y_global = y_0 + (y_1 - y_0)*(x_global - x_0)/(x_1 - x_0)
        #print(x_global, y_global, z_global)
        return y_global, z_global

    def intersectionY(self,x_0, y_0, z_0, x_1, y_1, z_1, y_global):
        #find intersection point
        #z_global = 0
        z_global = z_0 + (z_1 - z_0)*(y_global - y_0)/(y_1 - y_0)
        x_global = x_0 + (x_1 - x_0)*(y_global - y_0)/(y_1 - y_0)
        #print(x_global, y_global, z_global)
        return x_global, z_global

    def pixcel2global(self,px, py, z_global, camPos, verify = True):
        x_0, y_0, z_0, x_1, y_1, z_1 = self.pixcel2ray(px, py, camPos)
        x_global, y_global = self.intersectionZ(x_0, y_0, z_0, x_1, y_1, z_1, z_global = z_global)
        if verify:
            #verify
            #global to camera coordinate
            xc = self.matR[camPos][0][0]*x_global + self.matR[camPos][0][1]*y_global + self.matR[camPos][0][2]*z_global + self.vecT[camPos][0]
            yc = self.matR[camPos][1][0]*x_global + self.matR[camPos][1][1]*y_global + self.matR[camPos][1][2]*z_global + self.vecT[camPos][1]
            zc = self.matR[camPos][2][0]*x_global + self.matR[camPos][2][1]*y_global + self.matR[camPos][2][2]*z_global + self.vecT[camPos][2]

            #print(xc, yc, zc)
            len2D = math.sqrt(xc*xc + yc*yc)
            len3D = math.sqrt(xc*xc + yc*yc + zc*zc)
            if len2D > 0:
                theta = math.asin(len2D/len3D)
                thetaSq = theta*theta
                cdist = theta * (1.0 +  self.matD[camPos][0] * thetaSq + 
                                        self.matD[camPos][1] * thetaSq * thetaSq +
                                        self.matD[camPos][2] * thetaSq * thetaSq * thetaSq +
                                        self.matD[camPos][3] * thetaSq * thetaSq * thetaSq * thetaSq)
                xd = xc * cdist/len2D
                yd = yc * cdist/len2D
                vecPoint2D_u = int(self.matK[camPos][0] * xd + self.matK[camPos][1])
                vecPoint2D_v = int(self.matK[camPos][2] * yd + self.matK[camPos][3])
            #vecPoint2D_u, vecPoint2D_v must be similar with px py
            # print(vecPoint2D_u, vecPoint2D_v, vecPoint2D_u-px, vecPoint2D_v - py)
        return (x_global/self.dX, y_global/self.dY)
        # return (x_global, y_global)

    def heigh_estimation_X(self, px, py, x_global, camPos):
        x_0, y_0, z_0, x_1, y_1, z_1 = self.pixcel2ray(px, py, camPos)
        y_global, z_global = self.intersectionX(x_0, y_0, z_0, x_1, y_1, z_1, x_global = x_global)
        return z_global

    def heigh_estimation_Y(self, px, py, y_global, camPos):
        x_0, y_0, z_0, x_1, y_1, z_1 = self.pixcel2ray(px, py, camPos)
        x_global, z_global = self.intersectionY(x_0, y_0, z_0, x_1, y_1, z_1, y_global = y_global)
        return z_global


    def precompute_uvmap(self, grid_rows, grid_cols, cam_data):
        camera_extrinsics = np.identity(4)
        matrixR = np.array(cam_data['matrixR']).reshape(3,3)
        camera_extrinsics[:3, :3] = matrixR
        camera_extrinsics[:3, 3] = np.array(cam_data['vectT'])
        # print(camera_extrinsics, cam_data['vectT'])
        matrixK = cam_data['matrixK']
        camera_intrinsics_k_mat =  np.array([[matrixK[0], 0.0, matrixK[1]], [0.0, matrixK[2], matrixK[3]], [0.0, 0.0, 1.0]])
        matrixD = cam_data['matrixD']
        camera_intrinsics_d_mat = np.array(matrixD)

        ## new code
        self.K = camera_intrinsics_k_mat
        self.D = camera_intrinsics_d_mat
        # new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(camera_intrinsics_k_mat, camera_intrinsics_d_mat, 
        #                                         (self.fisheye_shape[0], self.fisheye_shape[1]), np.eye(3), balance=1)
        # mapping, map2 = cv2.fisheye.initUndistortRectifyMap(camera_intrinsics_k_mat, camera_intrinsics_d_mat, np.eye(3), 
        #                                         new_K, (self.fisheye_shape[0], self.fisheye_shape[1]), cv2.CV_16SC2)
        ##
        x = np.arange(0, grid_cols)
        y = np.arange(0, grid_rows)
        xv, yv = np.meshgrid(x, y)
        zv = np.ones_like(xv)
        bev_idx = np.stack([xv, yv, zv], axis=0).reshape(3, -1)
        # print('bev_image_idx', bev_idx.shape, bev_idx[:,0])
        world_idx = np.matmul(self.K_G_inv, bev_idx)
        # print('world_idx_2d', world_idx.shape, world_idx[:,0])
        world_idx_3d = np.zeros([4, world_idx.shape[1]])
        world_idx_3d[:2] = world_idx[:2]
        world_idx_3d[3] = world_idx[2]
        # print('world_idx_3d', world_idx_3d.shape, world_idx_3d[:,0])
        camera_idx = np.matmul(camera_extrinsics, world_idx_3d)
        # print('extrinsic_idx', camera_idx.shape, camera_idx[:,0])
        camera_idx = np.stack([camera_idx[0] / camera_idx[2], camera_idx[1] / camera_idx[2]])
        # print('camera_idx', camera_idx.shape, camera_idx[:,0])
        fisheye_idx_ = cv2.fisheye.distortPoints(np.transpose(camera_idx, (1,0))[None], K= camera_intrinsics_k_mat, D=camera_intrinsics_d_mat)
        fisheye_idx = fisheye_idx_[0].reshape(grid_rows, grid_cols, 2)
        # print('fisheye_idx', fisheye_idx.shape, fisheye_idx[0, 0])
        fisheye_uvmap = fisheye_idx.astype(np.float32)
        
        return fisheye_uvmap, fisheye_idx_[0], world_idx
    

    def unpack_data(self, data):
        images = [None, None, None, None]
        images[self.CAM_FRONT] = data['front']
        images[self.CAM_LEFT] = data['left']
        images[self.CAM_RIGHT] = data['right']
        images[self.CAM_REAR] = data['rear']
        return images

    def handle(self, data):
        fe_images = self.unpack_data(data)
        bev_images = [cv2.remap(im, fisheye_uvmap, None, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=self.borderValue) 
                                                for im, fisheye_uvmap in zip(fe_images, self.uv_maps)]
        self.bev_images=bev_images
        topview = np.zeros(self.topview_size, dtype=fe_images[0].dtype)
        topview[self.bev_alphas[self.CAM_FRONT]] = bev_images[self.CAM_FRONT][self.bev_alphas[self.CAM_FRONT]]
        topview[self.bev_alphas[self.CAM_REAR]] = bev_images[self.CAM_REAR][self.bev_alphas[self.CAM_REAR]]
        topview[self.bev_alphas[self.CAM_LEFT]] = bev_images[self.CAM_LEFT][self.bev_alphas[self.CAM_LEFT]]
        topview[self.bev_alphas[self.CAM_RIGHT]] = bev_images[self.CAM_RIGHT][self.bev_alphas[self.CAM_RIGHT]]
        # topview = np.flip(topview, 0)
        return topview

    def extrapolate_uvmap(self):
        topview_shape = (self.topview_size[:2][1], self.topview_size[:2][0]) #x,y
        outputs = []
        # import ipdb; ipdb.set_trace()
        # for i in range(int(self.fisheye_shape[0])):
        #     xindex_range = int(i* (topview_shape[0]/self.fisheye_shape[0]))
        #     ulist = self.uv_maps[2][:,:,0][:,xindex_range:xindex_range+1][0].tolist()
        #     xvalue = take_closest(ulist, i)
        #     t_ulist =  self.world_idxs[2][:,:,0][:,xindex_range:xindex_range+1][0]
        #     t_xvalue = t_ulist[ulist.index(xvalue)]
        #     for j in range(int(self.fisheye_shape[1])):
        #         yindex_range = int(j* (topview_shape[1]/self.fisheye_shape[1]))
        #         vlist = self.uv_maps[2][:,:,0][yindex_range:yindex_range+1,:][0].tolist()
        #         y_value = take_closest(vlist, j)      
        #         t_vlist =  self.world_idxs[2][:,:,0][yindex_range:yindex_range+1,:][0]
        #         t_yvalue = t_vlist[vlist.index(y_value)]
        #         outputs.append([i,j,t_xvalue*10 + 80, t_yvalue*10 + 90])
        return np.array(outputs)

    def process(self, rear_img, track_im, track_data, track_ids, frame_id):
        rear_topv = cv2.remap(rear_img, self.uv_maps[2],None, 
                        interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        # self.topview[self.bev_alphas[self.CAM_REAR]]  = rear_topv[self.bev_alphas[self.CAM_REAR]]

        for data, obj_id in zip(track_data, track_ids):
            x1, y1, w, h = data
            obj_id = int(obj_id)
            color = color = get_color(abs(obj_id)) #(0, 0, 255)
            # center bottom, *2 to map with original fisheye shape 
            X = (x1 + w/2) * 2 
            Y = (y1 + h) * 2
            
            # map_x = np.array([X], dtype=np.float32)
            # map_y = np.array([Y], dtype=np.float32)
            # track_pos_color = cv2.remap(src=rear_img, dst=rear_topv, map1=map_x, map2=map_y,
            #                 interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,borderValue=0)
            # r=0
            # g=1
            # b=2
            # r_query = track_pos_color[0][0][0]
            # g_query = track_pos_color[0][0][1]
            # b_query = track_pos_color[0][0][2]
            # center_coordinates = np.where((rear_topv[:,:,r] == r_query) & (rear_topv[:,:,g] == g_query) & (rear_topv[:,:,b] == b_query))
            # try:
            #     if int(center_coordinates[1]) < 85:
            #         self.topview = cv2.circle(self.topview, (int(center_coordinates[1]),int(center_coordinates[0])), radius=5, color=color, thickness=-1)
            #     else:
            #         print(f"X: {X}, Y: {Y}")
            #         print(f"Coords 1: {int(center_coordinates[1])}, Coords 0: {int(center_coordinates[0])}")
            # except:
            #     pass
            test_p = np.array([X, Y])
            center_coordinates_x, center_coordinates_y = self.pixcel2global(test_p[0], test_p[1], z_global = 0.0, camPos = 2, verify=False)
            center_coordinates_x = 80 + center_coordinates_x
            center_coordinates_y = 90 - center_coordinates_y

            self.topview = cv2.circle(self.topview, (int(center_coordinates_x), int(center_coordinates_y)), radius=2, color=color, thickness=-1)

            if obj_id not in self.track_data.keys():
                forward = 0
                self.track_data[obj_id] = [[center_coordinates_x, center_coordinates_y, forward]]
                self.track_objs[obj_id] = {"kf": KalmanFilter()}
                self.track_objs[obj_id].update({"meanval": self.track_objs[obj_id]['kf'].initiate([center_coordinates_x, center_coordinates_y, 1, 1])})

            else:
                # if (center_coordinates_x - self.track_data[obj_id][-1][0])  >= 0.0:
                #     forward = 1
                # else:
                #     forward = 0
                
                mean, variance = self.track_objs[obj_id]['kf'].predict(self.track_objs[obj_id]["meanval"][0], self.track_objs[obj_id]["meanval"][1])
                if mean[0] - self.track_objs[obj_id]["meanval"][0][0] >=0:
                    forward = 1
                else:
                    forward = 0
                self.track_objs[obj_id]['meanval'] = self.track_objs[obj_id]['kf'].update(mean, variance, [center_coordinates_x, center_coordinates_y, 1, 1])

            text_scale = 1
            text_thickness = 2
            line_thickness = 2
            intbox = tuple(map(int, (x1, y1+20, x1 + w, y1 + h)))
            id_text = f"F: {bool(forward)}"
            if forward:
                track_im =  cv2.putText(track_im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (255, 0, 0), thickness=text_thickness)
            else:
                track_im =  cv2.putText(track_im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 255, 0), thickness=text_thickness)

        return self.topview, track_im


if __name__=="main":
    with open(f'calib/cameraData.json', 'r') as f:
        calib_data = json.load(f)
        calib_data = calib_data['Items']

    bev_mapping = TopView360Handler(calib_data, num_channels=3, dX=0.1, dY = 0.1, grid_rows=180,
                                        grid_cols=160, borderValue=0)
    data_rgb = {
        "front": cv2.imread("/data/parking/code/parking/samples/data/image_1_1666166760959.png"),
        "rear": cv2.imread("/data/parking/code/parking/samples/data/image_0_1666166760836.png"),
        "left": cv2.imread("/data/parking/code/parking/samples/data/image_2_1666166760983.png"),
        "right": cv2.imread("/data/parking/code/parking/samples/data/image_3_1666166760935.png")
        }
        
    topview_image_rgb = bev_mapping.handle(data_rgb)
    cv2.imwrite("/data/parking/code/parking/samples/data/image_topview.png", 
                cv2.rotate(cv2.resize(topview_image_rgb, None, fx=2, fy=2, interpolation = cv2.INTER_CUBIC),cv2.ROTATE_90_CLOCKWISE)
                )