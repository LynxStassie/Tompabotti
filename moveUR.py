# move UR to initial pose - fixed position
# run detection code
# run transform code
# approach tomato
# pick tomato
# move to initial pose
# put tomato o the cart - fixed pose
# move to initial pose
# repeat until all tomatoes are picked - for loop
## detect code again - verification of tomatoes
# move the cart to the next position - fixed pose

# import libraries
import sys
import numpy as np
import urx
import time
import urx
#from main_defect_color import *

###================transform array to vector for UR========================###

# import scipy for transformation
from scipy.spatial.transform import Rotation as R # for rotation matrix to euler angles conversion

# counter-clockwise rotation of 90 degrees about the z-axis. This corresponds to the following quaternion (in scalar-last format)
#r = R.from_quat([0, 0, np.sin(np.pi/4), np.cos(np.pi/4)]) # rotation matrix for 45 degrees around z-axis (camera coordinate system) to robot coordinate system (UR5) conversion (UR5 coordinate system is rotated 45 degrees around z-axis)
# enter rotation array from terminal (ROS)
r = R.from_matrix([[0.0084493, -0.03262474, 0.99943196],
                   [0.99995635, -0.00371127, -0.00857488],
                   [0.00398892, 0.99946078, 0.03259196]])

# get vector for rx, ry, rz
r.as_rotvec()
print("as rotvec: ", r.as_rotvec())

# r.as_matrix()
# print("as matrix: ", r.as_matrix())

# r.as_euler('zyx', degrees=True)
# print(r.as_euler('zyx', degrees=True))

# r.as_quat()
# print(r.as_quat())


#
# # move UR to initial pose - fixed position
# rob = urx.Robot("192.168.56.8")
# #rob.set_tcp((0, 0, 0.1, 0, 0, 0))
# #rob.set_payload(2, (0, 0, 0.1))
# time.sleep(0.2)  #leave some time to robot to process the setup commands
# basket_position_relative_robot = [0.5, 0.5, 0.5, 0, 0, 0]
# if input('ROBOT READY. Press y to continue...') == 'y':
#     cartesian_intial = rob.get_pose()
#     print("cartesian_initial= ", cartesian_intial)
#     x_initial = cartesian_intial[0]
#     y_initial = cartesian_intial[1]
#     z_initial = cartesian_intial[2]
#
#     orientation_initial = rob.get_orientation()
#     print("orientation_initial= ", orientation_initial)
#     rx_initial = orientation_initial[0]
#     ry_initial = orientation_initial[1]
#     rz_initial = orientation_initial[2]
#
#     #rob.movej(x_initial, y_initial, z_initial, rx_initial, ry_initial,rz_initial)
#     #rob.movel((x, y, z, rx, ry, rz), a, v)
#
#     # run detection code
#     #os.system('python3 main_defect_color.py')
#
#     #print("Detection done")
#
#     # run transform code
#     #os.system('python3 TRANSFROM_TO_ADD.py')
#
#     # open gripper
#     rob.gripper.open()
#
#     # for i in range(len(TRANSFORMED_COORDS)):
#     #     x = TRANSFORMED_COORDS[i][0]
#     #     y = TRANSFORMED_COORDS[i][1]
#     #     z = TRANSFORMED_COORDS[i][2]
#     orientation = rob.get_orientation()
#     rx = orientation[3]
#     ry = orientation[4]
#     rz = orientation[5]
#
#     # approach tomato
#     rob.movel(x, y, z, rx, ry, rz)
#     rob.translate((x, y, z), wait=True) # towards tomato
#
#     # close gripper
#     rob.gripper.close()
#
#     # pick tomato
#     rob.translate(x, y, z) # picked tomato
#     rob.movel((x, y, z, rx, ry, rz), relative=True)
#
#     # move to initial pose
#     rob.movel(x_initial, y_initial, z_initial, rx_initial, ry_initial, rz_initial)
#     rob.movel(basket_position_relative_robot) # to the basket
#
#     # open gripper
#     rob.gripper.open()
#
#     # to the initial pose
#     rob.movel(x_initial, y_initial, z_initial, rx_initial, ry_initial, rz_initial)
# # repeat until all tomatoes are picked - for loop
#
#
# # move cart
# #
# # else:
# #       print('Set Robot to initial pose, kiitos')