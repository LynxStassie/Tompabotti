import pyrealsense2 as rs
import numpy as np
import cv2

# Start streaming

# Create a pipeline
pipeline = rs.pipeline()
config = rs.config()

config.enable_device_from_file("../realsense_d455.bag")
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)
# Getting the depth sensor's depth scale (see rs-align example for explanation)
# depth_sensor = profile.get_device().first_depth_sensor()
# depth_scale = depth_sensor.get_depth_scale()
# print("Depth Scale is: " , depth_scale)

depth_sensor = profile.get_device().first_depth_sensor()
auto_expl = depth_sensor.get_option(rs.option.enable_auto_exposure)
print("ae=",auto_expl)
auto_expl = 2.0
laser_pwr = depth_sensor.get_option(rs.option.laser_power)
print("laser power = ", laser_pwr)
# print(depth_sensor.get_option_range(rs.option.min_distance))
# print("laser power range = " , laser_range.min , "~", laser_range.max)
set_laser = 8
# depth_sensor.set_option(rs.option.min_distance, 0.25)
depth_scale = depth_sensor.get_depth_scale()
print(depth_sensor.get_depth_scale())
title = 'mouse event'

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        # depth_image_3d = np.dstack((depth_image,depth_image,depth_image))
        # print(np.shape(depth_image))
        # print(np.shape(color_image))
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.8), cv2.COLORMAP_JET)
        # depth_colormap_3d = np.dstack((depth_colormap,depth_colormap,depth_colormap))
        # depth_colormap = cv2.cvtColor(np.float32(depth_colormap))
                # cv2.imshow(title, color_frame)
        # cv2.imshow(title,color_image)
        cv2.imshow(title,color_image)

       #
        def onMouse(event, x, y, flags, params):
            if event == cv2.EVENT_RBUTTONDOWN:
                depth = depth_image[y,x].astype(float)
                print(y,x)
                distance = depth * depth_scale
                print ("Distance (m): ", distance)

        cv2.setMouseCallback(title,onMouse)
        cv2.imshow(title, depth_colormap)
        key  = cv2.waitKey(10)
        if key == 27:
            cv2.imwrite('../color_sensor.jpg',color_image)
            cv2.imwrite('../depth_sensor.jpg',depth_colormap)
            break
finally:
    cv2.destroyAllWindows()