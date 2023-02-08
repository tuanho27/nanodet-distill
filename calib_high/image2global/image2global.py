from cmath import pi
import numpy as np
import math
import json
import cv2 as cv

Camera_Pos = 2

#read calib data from file
def ReadCameraData(path):
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
def distort(theta, mat_D):
    thetaSq = theta*theta
    cdist = theta * (1.0 + mat_D[0] * thetaSq + 
                                mat_D[1] * thetaSq * thetaSq +
                                mat_D[2] * thetaSq * thetaSq * thetaSq +
                                mat_D[3] * thetaSq * thetaSq * thetaSq * thetaSq)
    return cdist

def pixcel2ray(px, py, camPos):
    #pixcel to sensor coordinate
    xd = float(px - matK[camPos][1])/matK[camPos][0]
    yd = float(py - matK[camPos][3])/matK[camPos][2]
    cdist = math.sqrt(xd*xd + yd*yd)
    #theta solver
    y = cdist
    cont = 1
    a = 0.0
    b = 3.14/2
    fa = distort(a, matD[camPos]) - y
    fb = distort(b, matD[camPos]) - y
    while cont > 0:
        x = a - (b-a)*fa/(fb-fa)
        fx = distort(x, matD[camPos]) - y
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
    x_0 = matR_inv[camPos][0][0]*(-vecT[camPos][0]) + matR_inv[camPos][0][1]*(-vecT[camPos][1]) + matR_inv[camPos][0][2]*(-vecT[camPos][2])
    y_0 = matR_inv[camPos][1][0]*(-vecT[camPos][0]) + matR_inv[camPos][1][1]*(-vecT[camPos][1]) + matR_inv[camPos][1][2]*(-vecT[camPos][2])
    z_0 = matR_inv[camPos][2][0]*(-vecT[camPos][0]) + matR_inv[camPos][2][1]*(-vecT[camPos][1]) + matR_inv[camPos][2][2]*(-vecT[camPos][2])

    #change second point to global coordinate
    x_1 = matR_inv[camPos][0][0]*(xc-vecT[camPos][0]) + matR_inv[camPos][0][1]*(yc-vecT[camPos][1]) + matR_inv[camPos][0][2]*(zc-vecT[camPos][2])
    y_1 = matR_inv[camPos][1][0]*(xc-vecT[camPos][0]) + matR_inv[camPos][1][1]*(yc-vecT[camPos][1]) + matR_inv[camPos][1][2]*(zc-vecT[camPos][2])
    z_1 = matR_inv[camPos][2][0]*(xc-vecT[camPos][0]) + matR_inv[camPos][2][1]*(yc-vecT[camPos][1]) + matR_inv[camPos][2][2]*(zc-vecT[camPos][2])
    #print(x_1,y_1,z_1)
    return x_0, y_0, z_0, x_1, y_1, z_1

def intersectionZ(x_0, y_0, z_0, x_1, y_1, z_1, z_global):
    #find intersection point
    #z_global = 0
    x_global = x_0 + (x_1 - x_0)*(z_global - z_0)/(z_1 - z_0)
    y_global = y_0 + (y_1 - y_0)*(z_global - z_0)/(z_1 - z_0)
    #print(x_global, y_global, z_global)
    return x_global, y_global

def intersectionX(x_0, y_0, z_0, x_1, y_1, z_1, x_global):
    #find intersection point
    #z_global = 0
    z_global = z_0 + (z_1 - z_0)*(x_global - x_0)/(x_1 - x_0)
    y_global = y_0 + (y_1 - y_0)*(x_global - x_0)/(x_1 - x_0)
    #print(x_global, y_global, z_global)
    return y_global, z_global

def intersectionY(x_0, y_0, z_0, x_1, y_1, z_1, y_global):
    #find intersection point
    #z_global = 0
    z_global = z_0 + (z_1 - z_0)*(y_global - y_0)/(y_1 - y_0)
    x_global = x_0 + (x_1 - x_0)*(y_global - y_0)/(y_1 - y_0)
    #print(x_global, y_global, z_global)
    return x_global, z_global

def pixcel2global(px, py, z_global, camPos, verify = True):
    x_0, y_0, z_0, x_1, y_1, z_1 = pixcel2ray(px, py, camPos)
    x_global, y_global = intersectionZ(x_0, y_0, z_0, x_1, y_1, z_1, z_global = z_global)
    if verify:
        #verify
        #global to camera coordinate
        xc = matR[camPos][0][0]*x_global + matR[camPos][0][1]*y_global + matR[camPos][0][2]*z_global + vecT[camPos][0]
        yc = matR[camPos][1][0]*x_global + matR[camPos][1][1]*y_global + matR[camPos][1][2]*z_global + vecT[camPos][1]
        zc = matR[camPos][2][0]*x_global + matR[camPos][2][1]*y_global + matR[camPos][2][2]*z_global + vecT[camPos][2]

        #print(xc, yc, zc)
        len2D = math.sqrt(xc*xc + yc*yc)
        len3D = math.sqrt(xc*xc + yc*yc + zc*zc)
        if len2D > 0:
            theta = math.asin(len2D/len3D)
            thetaSq = theta*theta
            cdist = theta * (1.0 +  matD[camPos][0] * thetaSq + 
                                    matD[camPos][1] * thetaSq * thetaSq +
                                    matD[camPos][2] * thetaSq * thetaSq * thetaSq +
                                    matD[camPos][3] * thetaSq * thetaSq * thetaSq * thetaSq)
            xd = xc * cdist/len2D
            yd = yc * cdist/len2D
            vecPoint2D_u = int(matK[camPos][0] * xd + matK[camPos][1])
            vecPoint2D_v = int(matK[camPos][2] * yd + matK[camPos][3])
        #vecPoint2D_u, vecPoint2D_v must be similar with px py
        print(vecPoint2D_u, vecPoint2D_v, vecPoint2D_u-px, vecPoint2D_v - py)
    return (x_global, y_global)

def heigh_estimation_X(px, py, x_global, camPos):
    x_0, y_0, z_0, x_1, y_1, z_1 = pixcel2ray(px, py, camPos)
    y_global, z_global = intersectionX(x_0, y_0, z_0, x_1, y_1, z_1, x_global = x_global)
    return z_global

def heigh_estimation_Y(px, py, y_global, camPos):
    x_0, y_0, z_0, x_1, y_1, z_1 = pixcel2ray(px, py, camPos)
    x_global, z_global = intersectionY(x_0, y_0, z_0, x_1, y_1, z_1, y_global = y_global)
    return z_global

#camera instrinsics & extrinsics
matD, matK, matR, vecT, matR_inv = ReadCameraData('./calib/cameraData.json')

# Read Image
image = cv.imread('./calib/img' + str(Camera_Pos) +'.png')
window_name = 'Image'
radius = 2
color = (255, 0, 0)
thickness = 1
font = cv.FONT_HERSHEY_SIMPLEX
fontScale = 0.5

if Camera_Pos == 2:
    #points in images
    image_points = np.array([ [327, 402], [218, 400], [452, 310], [339, 318],[957,407],[1065,405],[836,312],[948,325]])
    (m,n) = image_points.shape
    image_global_points = np.zeros((m, 2))
    for i in range(m):
        image_global_points[i] = pixcel2global(image_points[i][0], image_points[i][1] , z_global = 0.0, camPos = Camera_Pos)

    #show result

    for i in range(m):
        image = cv.circle(image, image_points[i], radius, (0,0,255), 2)
        output = str(round(image_global_points[i][0],2)) + ", " + str(round(image_global_points[i][1],2))
        image = cv.putText(image, output, image_points[i],font, fontScale, color, thickness, cv.LINE_AA)

    #height estimate
    #xe fadil
    test_p = np.array([702, 214])
    test_h = np.array([704, 193])
    test_x, test_y = pixcel2global(test_p[0], test_p[1], z_global = 0.0, camPos = Camera_Pos)
    test_z = heigh_estimation_X(test_h[0], test_h[1], x_global = test_x, camPos = Camera_Pos)
    print(test_z)
    test_z = heigh_estimation_Y(test_h[0], test_h[1], y_global = test_y, camPos = Camera_Pos)
    print(test_z)
    image = cv.circle(image, test_h, radius, (0,0,255), 2)
    image = cv.circle(image, test_p, radius, (0,0,255), 2)
    output = str(round(test_z,3))
    image = cv.putText(image, output, test_h,font, fontScale, color, thickness, cv.LINE_AA)

    #xe e34 trang
    test_p = np.array([572, 230])
    test_h = np.array([560, 169])
    test_x, test_y = pixcel2global(test_p[0], test_p[1], z_global = 0.0, camPos = Camera_Pos)
    test_z = heigh_estimation_X(test_h[0], test_h[1], x_global = test_x, camPos = Camera_Pos)
    print(test_z)
    test_z = heigh_estimation_Y(test_h[0], test_h[1], y_global = test_y, camPos = Camera_Pos)
    print(test_z)
    image = cv.circle(image, test_h, radius, (0,0,255), 2)
    image = cv.circle(image, test_p, radius, (0,0,255), 2)
    output = str(round(test_z,3))
    image = cv.putText(image, output, test_h,font, fontScale, color, thickness, cv.LINE_AA)

    #xe e34 xanh
    test_p = np.array([680, 232])
    test_h = np.array([680, 159])
    test_x, test_y = pixcel2global(test_p[0], test_p[1], z_global = 0.0, camPos = Camera_Pos)
    test_z = heigh_estimation_X(test_h[0], test_h[1], x_global = test_x, camPos = Camera_Pos)
    print(test_z)
    test_z = heigh_estimation_Y(test_h[0], test_h[1], y_global = test_y, camPos = Camera_Pos)
    print(test_z)
    image = cv.circle(image, test_h, radius, (0,0,255), 2)
    image = cv.circle(image, test_p, radius, (0,0,255), 2)
    output = str(round(test_z,3))
    image = cv.putText(image, output, test_h,font, fontScale, color, thickness, cv.LINE_AA)

if Camera_Pos == 0:
    # edge table 1 left
    test_p = np.array([760, 294])
    test_h = np.array([784, 159])
    test_x, test_y = pixcel2global(test_p[0], test_p[1], z_global = 0.0, camPos = Camera_Pos)
    test_z = heigh_estimation_X(test_h[0], test_h[1], x_global = test_x, camPos = Camera_Pos)
    print(test_z)
    #test_z = heigh_estimation_Y(test_h[0], test_h[1], y_global = test_y, camPos = Camera_Pos)
    #print(test_z)
    image = cv.circle(image, test_h, radius, (0,0,255), 2)
    image = cv.circle(image, test_p, radius, (0,0,255), 2)
    output = str(round(test_z,3))
    image = cv.putText(image, output, test_h,font, fontScale, color, thickness, cv.LINE_AA)

    # edge table 1 right
    test_p = np.array([824, 299])
    test_h = np.array([865, 174])
    test_x, test_y = pixcel2global(test_p[0], test_p[1], z_global = 0.0, camPos = Camera_Pos)
    test_z = heigh_estimation_X(test_h[0], test_h[1], x_global = test_x, camPos = Camera_Pos)
    print(test_z)
    #test_z = heigh_estimation_Y(test_h[0], test_h[1], y_global = test_y, camPos = Camera_Pos)
    #print(test_z)
    image = cv.circle(image, test_h, radius, (0,0,255), 2)
    image = cv.circle(image, test_p, radius, (0,0,255), 2)
    output = str(round(test_z,3))
    image = cv.putText(image, output, test_h,font, fontScale, color, thickness, cv.LINE_AA)

    # edge table 2 left
    test_p = np.array([836, 299])
    test_h = np.array([866, 214])
    test_x, test_y = pixcel2global(test_p[0], test_p[1], z_global = 0.0, camPos = Camera_Pos)
    test_z = heigh_estimation_X(test_h[0], test_h[1], x_global = test_x, camPos = Camera_Pos)
    print(test_z)
    #test_z = heigh_estimation_Y(test_h[0], test_h[1], y_global = test_y, camPos = Camera_Pos)
    #print(test_z)
    image = cv.circle(image, test_h, radius, (0,0,255), 2)
    image = cv.circle(image, test_p, radius, (0,0,255), 2)
    output = str(round(test_z,3))
    image = cv.putText(image, output, test_h,font, fontScale, color, thickness, cv.LINE_AA)

    # edge table 2 right
    test_p = np.array([882, 304])
    test_h = np.array([917, 226])
    test_x, test_y = pixcel2global(test_p[0], test_p[1], z_global = 0.0, camPos = Camera_Pos)
    test_z = heigh_estimation_X(test_h[0], test_h[1], x_global = test_x, camPos = Camera_Pos)
    print(test_z)
    #test_z = heigh_estimation_Y(test_h[0], test_h[1], y_global = test_y, camPos = Camera_Pos)
    #print(test_z)
    image = cv.circle(image, test_h, radius, (0,0,255), 2)
    image = cv.circle(image, test_p, radius, (0,0,255), 2)
    output = str(round(test_z,3))
    image = cv.putText(image, output, test_h,font, fontScale, color, thickness, cv.LINE_AA)


if Camera_Pos == 1:
    # edge table left
    test_p = np.array([102, 450])
    test_h = np.array([81, 336])
    test_x, test_y = pixcel2global(test_p[0], test_p[1], z_global = 0.0, camPos = Camera_Pos)
    test_z = heigh_estimation_X(test_h[0], test_h[1], x_global = test_x, camPos = Camera_Pos)
    print(test_z)
    #test_z = heigh_estimation_Y(test_h[0], test_h[1], y_global = test_y, camPos = Camera_Pos)
    #print(test_z)
    image = cv.circle(image, test_h, radius, (0,0,255), 2)
    image = cv.circle(image, test_p, radius, (0,0,255), 2)
    output = str(round(test_z,3))
    image = cv.putText(image, output, test_h,font, fontScale, color, thickness, cv.LINE_AA)


cv.imwrite('test_' + str(Camera_Pos) + '.png',image)
cv.imshow(window_name, image)
cv.waitKey(0)