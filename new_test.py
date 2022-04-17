import os
import cv2 as cv
import numpy as np
import keyboard

import pyrealsense2 as rs

import matplotlib.pyplot as plt
# Initialize the camera
pipeline = rs.pipeline()
config = rs.config()
rs.log_to_file(rs.log_severity.debug, file_path='./log.txt')
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
key = -1

profile = pipeline.start(config)
while True:
    frames = pipeline.wait_for_frames()
    # color sensor
    color_sensor = profile.get_device().query_sensors()[1]
    color_sensor.set_option(rs.option.enable_auto_exposure, 1)
    align = rs.align(rs.stream.color)
    frameset = align.process(frames)

    #color_frame = frames.get_color_frame()
    color_frame = frameset.get_color_frame()
    color = np.asanyarray(color_frame.get_data())



    depth_sensor = profile.get_device().query_sensors()[0]
    depth_sensor.set_option(rs.option.enable_auto_exposure, True)

    exposure_value = depth_sensor.get_option(rs.option.exposure)  # Get exposure
    gain_value = depth_sensor.get_option(rs.option.gain)  # Get exposure

    depth_frame = frameset.get_depth_frame()


    depth_image = np.asanyarray(depth_frame.get_data())

    colorizer = rs.colorizer()
    colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
    colorized_depthColorMap = cv.applyColorMap(cv.convertScaleAbs(colorized_depth, alpha=14), cv.COLORMAP_OCEAN)

    cv.putText(colorized_depth, str("exposure=" + str(exposure_value)), (50 - 30, 50 - 20), cv.FONT_HERSHEY_SIMPLEX, 1,
               (255, 255, 100), 2)
    cv.putText(colorized_depth, str("gain=" + str(gain_value)), (50 - 30, 100 - 20), cv.FONT_HERSHEY_SIMPLEX, 1,
               (100, 255, 100), 2)

    plt.imshow(color)#, alpha=0.6)
    #plt.imshow(depth_image, alpha=0.8)
    plt.imshow(colorized_depth, alpha=0.6)
    plt.show()
    key = cv.waitKey(1)