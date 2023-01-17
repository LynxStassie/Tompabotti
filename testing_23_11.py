# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
# profile = pipeline.start(config)
# profi = profile.get_stream(rs.stream.depth)
# intrinss = profi.as_video_stream_profile().get_intrinsics()
# print("test focal fx= ", intrinss.fx)
# print("test focal fy= ", intrinss.fy)
# print("test ppx  = ", intrinss.ppx)
# print("test ppy  = ", intrinss.ppy)

import urinterface as ur
import logging
import time
import unittest

from urinterface.robot_connection import RobotConnection
#from utils.robot_connection_mockup import DummySocket, DummyDashboardSocket, DummyRTDE
import numpy as np
import time
import cv2

# ur.RobotConnection.set_ip("192.168.56.8")
# ur.RobotConnection.set_port(50002)
# ur.RobotConnection.set_timeout(5)
# ur.RobotConnection.set_debug(True)
# ur.RobotConnection.set_simulation(True)
# ur.RobotConnection.connect()

