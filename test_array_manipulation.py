
import numpy as np
import sys
directory1 = 'C:/Users/123456/PycharmProjects/Tompabotti/Tomato_project/Program'
####################################################
# Andreas & Christian Library
path = sys.path.append(directory1 + '/Library/CameraLib')
print(path)
coordsArray = np.loadtxt('tomato_coords_list_sorted.txt', dtype=float)
cordinate_camera = []
cordinate_robot = []

for i in range(len(coordsArray)):
    dummy_array = []
    for j in range(4):
        B=[]
        B.append(coordsArray[i][j])
        dummy_array.append(B)
    cordinate_camera.append(dummy_array)
print(cordinate_camera[1])
print(np.shape(cordinate_camera))


'''for i in range(len(cordinate_camera)):
    matrix_robot= xyz_World_coordinates_method_3(T0_6, cordinate_camera[i], calibration_transformation_matrix1)
    cordinate_robot.append(matrix_robot)
camera_vals = [[0], [1], [2], [3]]
print(np.shape(camera_vals))
print(camera_vals[0])'''