
import math
import py_compile
import socket
import time
import numpy as np
import sys
import cv2
import os
from datetime import datetime

from numpy import asarray
from numpy import savetxt
#import pandas as pd
import cv2
import numpy as np
import math
import sys
#import Forward_kinematics_UR16 as FK
#Andreas & Christian Library

###############################

#directory = '/home/user/Desktop/Hand_Eye_Calibration-main/Data/' #Should be changed to your directory
directory1 = 'C:/Users/123456/PycharmProjects/Tompabotti/Tomato_project/Program'
####################################################
# Andreas & Christian Library
path = sys.path.append(directory1 + '/Library/CameraLib')
print(path)

import camera2robotcalib as cam
import DenavitHartenberg

####################################################

sys.path.append(directory1 + '/Library/config')
sys.path.append(directory1 + '/Library/modules_AJT_CHR')


#----------------------------------------------------------------------------------------
# UR library configuration for connecting to UR via ethernet
import robot
import robotconfig as rcfg
from src.ur.class_ur import UR



#------------------------------------------------------------------------------------------
# reading file extracting data and creating 3D array

def convrt_Txt_to_float_list(filename):
    with open(filename, 'r') as fh:
        data = fh.readlines()
        data_Array =[float(item) for item in data] #converts to a list of floating numbers
        return data_Array

def matrix_2_3d_array (variable):
    ranger = len(variable)
    n=int(ranger/3)
    matrix_in_3D = []
    for i in range (n):
        inner_array_3D = [[variable[i]],[variable[i+1]],[variable[i+2]]]
        matrix_in_3D.append(inner_array_3D)
    return matrix_in_3D

def matrix_2d_3d_array(variable):
    ranger = len(variable)
    n=int(ranger/3)
    matrix_in_3d = []

#-------------------------------------------------------------------------------------------

# data extraction from file to 3D array
'''rvecs_list = convrt_Txt_to_float_list(directory + 'rvecs.txt')
tvecs_list = convrt_Txt_to_float_list(directory + 'tvecs.txt')
robotrvecs_list = convrt_Txt_to_float_list(directory + 'robotrvecs.txt')
robottvecs_list = convrt_Txt_to_float_list(directory + 'robottvecs.txt')

rvecs = np.array(matrix_2_3d_array(rvecs_list))
tvecs = np.array(matrix_2_3d_array(tvecs_list))
robotrvecs = np.array(matrix_2_3d_array(robotrvecs_list))
robottvecs = np.array(matrix_2_3d_array(robottvecs_list))

#----------------------------------------------------------------------------------------------


#extrinsic parametrs calculation
rm1 ,tm1 = cv2.calibrateHandEye(robotrvecs ,robottvecs ,rvecs ,tvecs ,method=cv2.CALIB_HAND_EYE_PARK)

# print(np.linalg.inv(Rm))
homogenousmatrixplaceholder = np.identity(4)
homogenousmatrixplaceholder[0:3 ,0:3] = rm1
homogenousmatrixplaceholder[0:3 ,3] = tm1.ravel()
print("Park-Martin Calibration Matrix (1):")
print(homogenousmatrixplaceholder)
print("")

rm2,tm2 = cv2.calibrateHandEye(robotrvecs,robottvecs,rvecs,tvecs,method=cv2.CALIB_HAND_EYE_TSAI)

homogenousmatrixplaceholder = np.identity(4)
homogenousmatrixplaceholder[0:3,0:3] = rm2
homogenousmatrixplaceholder[0:3,3] = tm2.ravel()
print("Tsai Calibration Matrix (2):")
print(homogenousmatrixplaceholder)
#transform = open(path+"/transformation.txt","a")
#fileobject.write(homogenousmatrixplaceholder)
#fileobject.close()
print("")



rm3 ,tm3 = cv2.calibrateHandEye(robotrvecs ,robottvecs ,rvecs ,tvecs ,method=cv2.CALIB_HAND_EYE_HORAUD)

homogenousmatrixplaceholder = np.identity(4)
homogenousmatrixplaceholder[0:3 ,0:3] = rm3
homogenousmatrixplaceholder[0:3 ,3] = tm3.ravel()
print("Horaud Calibration Matrix (3):")
print(homogenousmatrixplaceholder)
print("")


rm4 ,tm4 = cv2.calibrateHandEye(robotrvecs ,robottvecs ,rvecs ,tvecs ,method=cv2.CALIB_HAND_EYE_DANIILIDIS)

homogenousmatrixplaceholder = np.identity(4)
homogenousmatrixplaceholder[0:3 ,0:3] = rm4
homogenousmatrixplaceholder[0:3 ,3] = tm4.ravel()
print("Daniilidis Calibration Matrix (4):")
print(homogenousmatrixplaceholder)
print("")

calibrationchoice = 2
if(calibrationchoice == 1):
    print("Park-Martin Method")
    cTee = np.identity(4)
    cTee[0:3 ,0:3] = rm1
    cTee[0:3 ,3] = tm1.ravel()
    print(cTee)

elif(calibrationchoice == 2):
    print("Tsai Method")
    cTee = np.identity(4)
    cTee[0:3 ,0:3] = rm2
    cTee[0:3 ,3] = tm2.ravel()
    print(cTee)

elif(calibrationchoice == 3):
    print("Horaud Method")
    cTee = np.identity(4)
    cTee[0:3 ,0:3] = rm3
    cTee[0:3 ,3] = tm3.ravel()
    print(cTee)

elif(calibrationchoice == 4):
    print("Daniilidis Method")
    cTee = np.identity(4)
    cTee[0:3 ,0:3] = rm4
    cTee[0:3 ,3] = tm4.ravel()

else:
    # default park-martin
    print("Default Park-Martin Method")
    cTee = np.identity(4)
    cTee[0:3 ,0:3] = rm1
    cTee[0:3 ,3] = tm1.ravel()
    print(cTee)

print("")

#print("Image Distortion Parameters:")
#print(dist)
#print("")

#print("Camera Intrinsic Matrix:")
#print(mtx)
#print("")

#print("Optimal Camera Intrinsic Matrix")
#print(newcameramtx)
#print("")
'''




#--------------------------------------------------------------------------------------------------
#Calculating inverse of a Homogeneous transformation matrix
def matrix_Inverse(transformation_matrix):
    inverse_Of_matrix = []
    rotation_Matrix_part = transformation_matrix[0:3,0:3]
    translation_Matrix_part = transformation_matrix[0:3,[3]]
    transpose_Of_Rotation_Matrix = np.transpose(rotation_Matrix_part)
    translation_Vector = np.array(-transpose_Of_Rotation_Matrix @ translation_Matrix_part)
    inverse_Of_matrix = np.concatenate ((np.concatenate((transpose_Of_Rotation_Matrix,translation_Vector),1),[[0,0,0,1]]),0)
    return inverse_Of_matrix

#---------------------------------------------------------------------------------------------------

# getting a pose value (x,y,z,rx,ry,rz) value from the robot
HOST1 = rcfg.HOST_IP
PORT1 = 30003  # The same port as used by the server
ur = UR(HOST1, PORT1)
pose_ur = ur.get_position(world=False)
print(pose_ur)

#--------------------------------------------------------------------------------------------------
# converting (rx,ry,rz) values to rotation matrix
def rotation_matrix(theta1, theta2, theta3, order='xyz'):
    """
    input
        theta1, theta2, theta3 = rotation angles in rotation order (degrees)
        oreder = rotation order of x,y,z　e.g. XZY rotation -- 'xzy'
    output
        3x3 rotation matrix (numpy array)
    """
    c1 = np.cos(theta1)# * np.pi / 180)
    s1 = np.sin(theta1 )#* np.pi / 180)
    c2 = np.cos(theta2)# * np.pi / 180)
    s2 = np.sin(theta2)# * np.pi / 180)
    c3 = np.cos(theta3)# * np.pi / 180)
    s3 = np.sin(theta3)# * np.pi / 180)

    if order == 'xzx':
        matrix=np.array([[c2, -c3*s2, s2*s3],
                         [c1*s2, c1*c2*c3-s1*s3, -c3*s1-c1*c2*s3],
                         [s1*s2, c1*s3+c2*c3*s1, c1*c3-c2*s1*s3]])
    elif order=='xyx':
        matrix=np.array([[c2, s2*s3, c3*s2],
                         [s1*s2, c1*c3-c2*s1*s3, -c1*s3-c2*c3*s1],
                         [-c1*s2, c3*s1+c1*c2*s3, c1*c2*c3-s1*s3]])
    elif order=='yxy':
        matrix=np.array([[c1*c3-c2*s1*s3, s1*s2, c1*s3+c2*c3*s1],
                         [s2*s3, c2, -c3*s2],
                         [-c3*s1-c1*c2*s3, c1*s2, c1*c2*c3-s1*s3]])
    elif order=='yzy':
        matrix=np.array([[c1*c2*c3-s1*s3, -c1*s2, c3*s1+c1*c2*s3],
                         [c3*s2, c2, s2*s3],
                         [-c1*s3-c2*c3*s1, s1*s2, c1*c3-c2*s1*s3]])
    elif order=='zyz':
        matrix=np.array([[c1*c2*c3-s1*s3, -c3*s1-c1*c2*s3, c1*s2],
                         [c1*s3+c2*c3*s1, c1*c3-c2*s1*s3, s1*s2],
                         [-c3*s2, s2*s3, c2]])
    elif order=='zxz':
        matrix=np.array([[c1*c3-c2*s1*s3, -c1*s3-c2*c3*s1, s1*s2],
                         [c3*s1+c1*c2*s3, c1*c2*c3-s1*s3, -c1*s2],
                         [s2*s3, c3*s2, c2]])
    elif order=='xyz':
        matrix=np.array([[c2*c3, -c2*s3, s2],
                         [c1*s3+c3*s1*s2, c1*c3-s1*s2*s3, -c2*s1],
                         [s1*s3-c1*c3*s2, c3*s1+c1*s2*s3, c1*c2]])
    elif order=='xzy':
        matrix=np.array([[c2*c3, -s2, c2*s3],
                         [s1*s3+c1*c3*s2, c1*c2, c1*s2*s3-c3*s1],
                         [c3*s1*s2-c1*s3, c2*s1, c1*c3+s1*s2*s3]])
    elif order=='yxz':
        matrix=np.array([[c1*c3+s1*s2*s3, c3*s1*s2-c1*s3, c2*s1],
                         [c2*s3, c2*c3, -s2],
                         [c1*s2*s3-c3*s1, c1*c3*s2+s1*s3, c1*c2]])
    elif order=='yzx':
        matrix=np.array([[c1*c2, s1*s3-c1*c3*s2, c3*s1+c1*s2*s3],
                         [s2, c2*c3, -c2*s3],
                         [-c2*s1, c1*s3+c3*s1*s2, c1*c3-s1*s2*s3]])
    elif order=='zyx':
        matrix=np.array([[c1*c2, c1*s2*s3-c3*s1, s1*s3+c1*c3*s2],
                         [c2*s1, c1*c3+s1*s2*s3, c3*s1*s2-c1*s3],
                         [-s2, c2*s3, c2*c3]])
    elif order=='zxy':
        matrix=np.array([[c1*c3-s1*s2*s3, -c2*s1, c1*s3+c3*s1*s2],
                         [c3*s1+c1*s2*s3, c1*c2, s1*s3-c1*c3*s2],
                         [-c2*s3, s2, c2*c3]])

    return matrix

#----------------------------------------------------------------------------------------------------
# getting Homogeneous Transformation Matrix from Pose valur obtained from robot P(x ,y, z, rx, ry, rz)
def pose_2_transformation_matrix(given_pose):

    #converting rx,ry,rz to rotation matrix
    R06 = rotation_matrix(given_pose[3],given_pose[4],given_pose[5], order='zxy')

    #translation vector
    D0_6 = [[given_pose[0]],[given_pose[1]],[given_pose[2]]]

    #concatenate translation and rotation
    T06=np.concatenate((np.concatenate((R06,D0_6),1),[[0,0,0,1]]),0)

    return T06


#---------------------------------------------------------------------------------------------------
#creating a transfomation matrix in 4 by 4 form
def creating_4X4_matrix (matrix_3X4):
    row = np.array([0,0,0,1])
    matrix_4x4 = np.append(matrix_3X4,[row],axis=0)
    return matrix_4x4


#----------------------------------------------------------------------------------------------------
# (method 1)
# getting world XYZ coordinates relative to robot base from pixel values using camera Intrinsic matrix
def xyz_World_coordinates_method_1(RobotT06, trans_cam_flange, Intrinsic_mtx, pixel_value):
    transformation_camera_relative_2_robobase = RobotT06 @ trans_cam_flange

    #creating a 3 X 4 camera intrinsic matrix adding 0
    new_column = [[0],[0],[0]]
    camera_Intrinsic_mtx= np.append(Intrinsic_mtx, new_column,axis=1)

    # the overall projection matrix is the product of Instrinic matrix with excentric matrix (homogeneous matrix of camera relative to robobase)
    projection_matrix = camera_Intrinsic_mtx @ transformation_camera_relative_2_robobase

    projection_T_parameters = (projection_matrix[0:3,3])
    projection_T_mtx = np.reshape((np.array(projection_T_parameters)),(3,1))
    new_row = [[1]]
    projection_T_mtx = np.append(projection_T_mtx, new_row,axis=0)

    # the projection matrix is in the form of 3 X 4 matrix so there is no inverse
    sq_projection_mtx = creating_4X4_matrix(projection_matrix)
    XYZ_value = matrix_Inverse(sq_projection_mtx) @ pixel_value #+ projection_T_mtx
    return XYZ_value


#     #------------------------#     #------------------------#     #------------------------
# (method 2)
# obtaining real world XYZ values relative to base of the robot from camera xyz values using inverse transformation matrix
def xyz_World_coordinates_method_2(RobotT06, camera_xyz_value, camera_extrinsic_mtx): #input Complete transformation matrix of ROBOT T06 and
    transformation_camera_relative_2_robobase = RobotT06 @ camera_extrinsic_mtx
    XYZ_values_for_move = matrix_Inverse(transformation_camera_relative_2_robobase) @ camera_xyz_value
    return XYZ_values_for_move
#     #------------------------#     #------------------------#     #------------------------

# (method 3)
#obtaining real world values relative to base of the robot from camera xyz values using direct multiplication T_base_camera * point
def xyz_World_coordinates_method_3(RobotT06, camera_xyz_value, camera_extrinsic_mtx): #input Complete transformation matrix of ROBOT T06 and
    transformation_camera_relative_2_robobase = RobotT06 @ camera_extrinsic_mtx
    XYZ_values_for_move =transformation_camera_relative_2_robobase @ camera_xyz_value
    return XYZ_values_for_move
#direct multiplication of object pose with camera calibration matrix
def xyz_World_coordinates_method_4(RobotT06,camera_xyz_value, camera_extrinsic_mtx): #input Complete transformation matrix of ROBOT T06 and
    camera_extrinsic_mtx_inverse = matrix_Inverse(np.array(camera_extrinsic_mtx))
    XYZ_values_for_move1 = (RobotT06 @ camera_extrinsic_mtx_inverse) @ camera_xyz_value
    return XYZ_values_for_move1

#----------------------------------------------------------------------------------------------------




# overall transformation matrix of robot EE relative to base using the pose value from the robot


# type 1# print('XYZ value obtained from DH parameter\n', xyz_World_coordinates_method_3(T0_6, camera_vals, calibration_transformation_matrix))

# using the pose value from the robot
T_0_6=(pose_2_transformation_matrix(pose_ur))
print('Transformation matrix from using the pose value from robot\n',T_0_6,'\n')

# type 2
# using the DH parameter values

jointposition = ur.get_joints_radv2()
theta = np.array(jointposition[0:6])
print(theta)
T0_6= np.array(DenavitHartenberg.DenavitHartenbergUR(theta, 'UR5e'))
print('Transformation matrix from DH parameter\n',T0_6,'\n')

#------------------------------------------------------------------------------------------------------


#pixel value from the camera
pixel_matrix = np.array([[254.99],[323.68],[1],[1]])

#-----------------------------------------------------------------------------------------------------

# X,Y,Z values received from camera
camera_vals =[[-44/1000],[-2/1000],[350/1000],[1]]
#camera_vals_1 = np.array([[-46],[59],[295],[1]])



calibration_transformation_matrix= [[-0.0371367,  -0.99920554,  0.01446194,  0.09525291], [ 0.99928393, -0.03702691,  0.00778672, -0.03834371], [-0.00724506,  0.01474076,  0.9998651,   0.07633219], [ 0,        0,         0,        1.        ]]
calibration_transformation_matrix1 = np.load('/home/user/Downloads/Hand_Eye_Calibration-main/Data/PARK_matrix.npy')
#------------------------------------------------------------------------------------------------------
print('transformation matrix from DH method\n',T0_6,'\n')
print('XYZ values from robot get_position\n', xyz_World_coordinates_method_3(T0_6, camera_vals, calibration_transformation_matrix1), '\n\n')
#print('XYZ value obtained from direct multiplication\n', xyz_World_coordinates_method_4(camera_vals_1, calibration_transformation_matrix))
#print('XYZ value obtained from direct inverse transform multiplication\n', xyz_World_coordinates_method_2(camera_vals_1, calibration_transformation_matrix))