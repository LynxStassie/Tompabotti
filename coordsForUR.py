# python file will read the detection list .txt and transform the coordinates to UR5's coordinate system.
# cameravals = [[x],[y],[z],[1]] -- numpy array
# libraries
import sys
from testing02 import *
import numpy as np
const = 1

def readCoordsFile(fileName):
    # open file for reading data
    # coordsArray = np.array([])
    coordsArray = np.loadtxt(fileName, dtype=float)
    # with open(fileName, 'r') as f:
    #     for line in f.readlines():         # read rest of lines
    #         # read from  item three elements of tuple to file
    #         x, y, z, const = line.split(' ')
    #         np.append(coordsArray,np.array([[float(x)], [float(y)], [float(z)], [float(const)]]))
    # # coordsArray.sort(key=lambda x: x[2])   # sort by third element of tuple (z) - distance from camera to tomato in mm
    return coordsArray

# def saveCoordsFileFromList(tomato_coords_list_of_tuples, fileName, mode):
#     print(tomato_coords_list_of_tuples)
#
#     # open file for ADDING data
#     with open(fileName, mode) as f:
#         for item in tomato_coords_list_of_tuples:
#             # write three elements of tuple to file
#             f.write("{} {} {} {}".format(item[0], item[1], item[2], item[3]))
#             f.write('\n')



sortedcoordslist = readCoordsFile('tomato_coords_list_sorted.txt')  # return listTuplesofCoords
print(sortedcoordslist) # print listTuplesofCoords
shape = sortedcoordslist.shape
print(shape)
# savesorted(listOfSorted, 'tomato_coords_list_sorted.txt', 'w')
