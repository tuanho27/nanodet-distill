from cmath import pi
import numpy as np
import math
import cv2 as cv
import json
from statistics import mean



def read_intrinsic(pathIntrinsic):
    f = open(pathIntrinsic, "r")
    Intrinsic = f.read()
    camera_matrix = Intrinsic.split("\n")[0].split(":")[1]
    dist_coefs = Intrinsic.split("\n")[1].split(":")[1]
    camera_matrix = json.loads(camera_matrix)
    dist_coefs = json.loads(dist_coefs)
    return (
        np.array(camera_matrix),
        np.array(dist_coefs)
    )

def read_extrinsic(pathExtrinsic):
    f = open(pathExtrinsic, "r")
    Extrinsic = f.read()
    rotation_matrix = Extrinsic.split("\n")[0].split(":")[1]
    translation_matrix = Extrinsic.split("\n")[1].split(":")[1]
    rotation_matrix = json.loads(rotation_matrix)
    translation_matrix = json.loads(translation_matrix)
    return (
        np.array(rotation_matrix),
        np.array(translation_matrix)
    )
def read_lane(pathSetPoints):
    f = open(pathSetPoints, "r")
    setPoints = f.read()
    left_points = setPoints.split("\n")[0].split(":")[1]
    right_points = setPoints.split("\n")[1].split(":")[1]
    left_points = json.loads(left_points)
    right_points = json.loads(right_points)
    return (
        np.array(left_points),
        np.array(right_points)
    )
#distort function
def distort(theta,matD):
    thetaSq = theta*theta
    cdist = theta * (1.0 + matD[0] * thetaSq + 
                                matD[1] * thetaSq * thetaSq +
                                matD[2] * thetaSq * thetaSq * thetaSq +
                                matD[3] * thetaSq * thetaSq * thetaSq * thetaSq)
    return cdist

def pixcel2global(px, py, matK, matD, matR, vecT, matR_inv, z_global  ):
    #pixcel to sensor coordinate
    xd = float(px - matK[0][2])/matK[0][0]
    yd = float(py - matK[1][2])/matK[1][1]
    cdist = math.sqrt(xd*xd + yd*yd)
    #theta solver
    y = cdist
    cont = 1
    a = 0.0
    b = 3.14/2
    fa = distort(a,matD) - y
    fb = distort(b,matD) - y
    while cont > 0:
        x = a - (b-a)*fa/(fb-fa)
        fx = distort(x,matD) - y
        if ( abs(fx) < 0.000001):
            cont = 0
        if (fx*fa < 0):
            b = x
        else:
            a = x
    #print(x, distort(x), distort(x) - y)
    theta = x
    len3D = 1/math.sin(theta)
    xc = xd/cdist
    yc = yd/cdist
    zc = math.sqrt(len3D*len3D - 1)
    #print(xc, yc, zc)

    #change camera position to global coordinate
    x_0 = matR_inv[0][0]*(-vecT[0]) + matR_inv[0][1]*(-vecT[1]) + matR_inv[0][2]*(-vecT[2])
    y_0 = matR_inv[1][0]*(-vecT[0]) + matR_inv[1][1]*(-vecT[1]) + matR_inv[1][2]*(-vecT[2])
    z_0 = matR_inv[2][0]*(-vecT[0]) + matR_inv[2][1]*(-vecT[1]) + matR_inv[2][2]*(-vecT[2])

    #change second point to global coordinate
    x_1 = matR_inv[0][0]*(xc-vecT[0]) + matR_inv[0][1]*(yc-vecT[1]) + matR_inv[0][2]*(zc-vecT[2])
    y_1 = matR_inv[1][0]*(xc-vecT[0]) + matR_inv[1][1]*(yc-vecT[1]) + matR_inv[1][2]*(zc-vecT[2])
    z_1 = matR_inv[2][0]*(xc-vecT[0]) + matR_inv[2][1]*(yc-vecT[1]) + matR_inv[2][2]*(zc-vecT[2])
    #print(x_1,y_1,z_1)

    #find intersection point
    #z_global = 0
    x_global = x_0 + (x_1 - x_0)*(z_global - z_0)/(z_1 - z_0)
    y_global = y_0 + (y_1 - y_0)*(z_global - z_0)/(z_1 - z_0)
    #print(x_global, y_global, z_global)

    #verify
    #global to camera coordinate
    xc = matR[0][0]*x_global + matR[0][1]*y_global + matR[0][2]*z_global + vecT[0]
    yc = matR[1][0]*x_global + matR[1][1]*y_global + matR[1][2]*z_global + vecT[1]
    zc = matR[2][0]*x_global + matR[2][1]*y_global + matR[2][2]*z_global + vecT[2]

    #print(xc, yc, zc)
    len2D = math.sqrt(xc*xc + yc*yc)
    len3D = math.sqrt(xc*xc + yc*yc + zc*zc)
    if len2D > 0:
        theta = math.asin(len2D/len3D)
        thetaSq = theta*theta
        cdist = theta * (1.0 +  matD[0] * thetaSq + 
                                matD[1] * thetaSq * thetaSq +
                                matD[2] * thetaSq * thetaSq * thetaSq +
                                matD[3] * thetaSq * thetaSq * thetaSq * thetaSq)
        xd = xc * cdist/len2D
        yd = yc * cdist/len2D
        vecPoint2D_u = int(matK[0][0] * xd + matK[0][2])
        vecPoint2D_v = int(matK[1][1] * yd + matK[1][2])
    #print(vecPoint2D_u, vecPoint2D_v)
    return (x_global, y_global)

def curve_approximation(points):
    (h,k) = points.shape
    matX = np.zeros((4, 4))
    vecY = np.zeros(4)
    for i in range(h):
        matX[0][0] += points[i][0]**6
        matX[0][1] += points[i][0]**5
        matX[0][2] += points[i][0]**4
        matX[0][3] += points[i][0]**3

        matX[1][0] += points[i][0]**5
        matX[1][1] += points[i][0]**4
        matX[1][2] += points[i][0]**3
        matX[1][3] += points[i][0]**2

        matX[2][0] += points[i][0]**4
        matX[2][1] += points[i][0]**3
        matX[2][2] += points[i][0]**2
        matX[2][3] += points[i][0]

        matX[3][0] += points[i][0]**3
        matX[3][1] += points[i][0]**2
        matX[3][2] += points[i][0]
        matX[3][3] += 1

        vecY[0] += points[i][0]**3 * points[i][1]
        vecY[1] += points[i][0]**2 * points[i][1]
        vecY[2] += points[i][0]**1 * points[i][1]
        vecY[3] += points[i][1]
    output = np.matmul(np.linalg.inv(matX),vecY)
    return output

def distance_p2c(curve, start_p, end_p, point):
    n_1st_sample = 200
    n_2nd_sample = 200
    xi = np.linspace(start_p[0], end_p[0], n_1st_sample + 1)
    yi = curve[0] + curve[1]*xi + curve[2]*xi*xi + curve[3]*xi*xi*xi
    di = []
    for i in range(len(xi)-1):
        distance_s = (xi[i] - point[0])*(xi[i] - point[0]) + (yi[i] - point[1])*(yi[i] - point[1])
        di.append(distance_s)

    min_ind  = di.index(min(di))
    start_p2 = [xi[min(min_ind - 1, 0)], yi[min(min_ind - 1, 0)]]
    end_p2   = [xi[max(min_ind + 1, n_1st_sample)], yi[max(min_ind + 1, n_1st_sample)]]

    xi2 = np.linspace(start_p2[0], end_p2[0], n_2nd_sample + 1)
    yi2 = curve[0] + curve[1]*xi2 + curve[2]*xi2*xi2 + curve[3]*xi2*xi2*xi2
    di2 = [] # distance^2 from point(x0, y0) to point (xi2, yi2) on curve
    for i in range(len(xi2)):
        distance_s = (xi2[i] - point[0])*(xi2[i] - point[0]) + (yi2[i] - point[1])*(yi2[i] - point[1])
        di2.append(distance_s)
        
    min_ind2  = di2.index(min(di2)) 
    return math.sqrt(min(di2))

def main():

    pathPoints = "./data/point.txt"
    pathIntrinsic = "./data/intrinsic.txt"
    pathExtrinsic = "./data/extrinsic.txt"
    pathImage  = "./data/image.png"
    pathOutput = "./data/visualResult.png"

    (left_points,right_points) = read_lane(pathPoints)
    (matK, matD) = read_intrinsic(pathIntrinsic)
    (matR, vecT) = read_extrinsic(pathExtrinsic)
    matR_inv = np.linalg.inv(matR)





    (m,n) = left_points.shape
    left_global_points = np.zeros((m, 2))
    for i in range(m):
        left_global_points[i] = pixcel2global(left_points[i][0], left_points[i][1] , matK,matD, matR, vecT,matR_inv, z_global = -2  )

    right_global_points = np.zeros((m, 2))
    for i in range(m):
        right_global_points[i] = pixcel2global(right_points[i][0], right_points[i][1] , matK,matD, matR, vecT,matR_inv, z_global = -2)


    curve_left = curve_approximation(left_global_points)
    curve_right = curve_approximation(right_global_points)
    # print(curve_left)
    # print(curve_right)

    # Reading an image in default mode
    image = cv.imread(pathImage)
    

    for i in range(m):
        image = cv.circle(image, left_points[i], 4, (0,0,255),-1, 1)
        output = str(round(-left_global_points[i][1],2)) + ", " + str(round(left_global_points[i][0],2))
        image = cv.putText(image, output, left_points[i],cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv.LINE_AA)
    for i in range(m):
        image = cv.circle(image, right_points[i], 4, (0,0,255),-1, 1)
        output = str(round(-right_global_points[i][1],2)) + ", " + str(round(right_global_points[i][0],2))
        image = cv.putText(image, output, right_points[i],cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv.LINE_AA)
    # cv.imwrite(pathOutput, image)
    curve = curve_left[::-1]

    listDistance = []
    for i in range(m):
        point = right_global_points[i] 
        xA = -10
        yA = curve[0] + curve[1]*xA + curve[2]*xA*xA + curve[3]*xA*xA*xA
        start_p = [xA, yA]
        xB = 10
        yB = curve[0] + curve[1]*xB + curve[2]*xB*xB + curve[3]*xB*xB*xB
        end_p   = [xB, yB]
        min_distance = distance_p2c(curve, start_p, end_p, point)
        listDistance.append(min_distance)
    lane_width = round(mean(listDistance),4)
    print('lane width ', lane_width)
    image = cv.putText(image, "lane width: "+ str(lane_width), (50,100),cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv.LINE_AA)
    cv.imwrite(pathOutput, image)

if __name__ == "__main__":
    main()
