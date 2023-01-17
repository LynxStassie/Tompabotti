# yolo with rotated bounding boxes

import numpy as np
import cv2
import time
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pyrealsense2 as rs
import imutils
from imutils.video import VideoStream

def get_rotated_bbox(bbox, angle):
    """
    Get rotated bounding box
    """
    x, y, w, h = bbox
    x_c, y_c = x + w/2, y + h/2
    x_c_rot, y_c_rot = x_c * np.cos(angle) - y_c * np.sin(angle), x_c * np.sin(angle) + y_c * np.cos(angle)
    w_rot, h_rot = w * np.cos(angle) + h * np.sin(angle), w * np.sin(angle) + h * np.cos(angle)
    x_rot, y_rot = x_c_rot - w_rot/2, y_c_rot - h_rot/2
    return x_rot, y_rot, w_rot, h_rot

def get_rotated_bbox_corners(bbox, angle):
    """
    Get rotated bounding box corners
    """
    x, y, w, h = bbox
    x_c, y_c = x + w/2, y + h/2
    x_c_rot, y_c_rot = x_c * np.cos(angle) - y_c * np.sin(angle), x_c * np.sin(angle) + y_c * np.cos(angle)
    w_rot, h_rot = w * np.cos(angle) + h * np.sin(angle), w * np.sin(angle) + h * np.cos(angle)
    x_rot, y_rot = x_c_rot - w_rot/2, y_c_rot - h_rot/2
    return np.array([[x_rot, y_rot], [x_rot + w_rot, y_rot], [x_rot + w_rot, y_rot + h_rot], [x_rot, y_rot + h_rot]])


# load yolo weights and config
tmodelConfiguration = os.path.join(os.path.dirname(__file__),"yolov3_training_rotated_bboxes.cfg") #label studio dataset and colab yolo training
tmodelWeights = os.path.join(os.path.dirname(__file__),"yolov3_training_rotated_bboxes.weights")
'''cv.dnn.readNetFromDarknet(tmodelConfiguration, tmodelWeights)'''
net = cv2.dnn.readNetFromDarknet(tmodelConfiguration,tmodelWeights)
# classes = []
with open("obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# initialize RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)
align_to = rs.stream.color
align = rs.align(align_to)


# initialize the video stream and allow the camera sensor to warmup
vs = VideoStream(src=0).start()
time.sleep(2.0) # allow camera to warmup

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it to have a maximum width of 500 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=500)
    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0,), True, crop=False)
    # set the blob as input to the network and perform a forward pass to get the output
    net.setInput(blob)
    outs = net.forward(output_layers)
    # initialize the lists of detected bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    classIDs = []
    # loop over the output layers from the network and add the bounding boxes, confidences, and class IDs to the appropriate lists
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.1:
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                cv2.circle(frame, (center_x, center_y), 4, (0, 255, 255), 5)
                cv2.putText(frame, "center...", (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                plt.imshow(frame)
                width = int(detection[2] * w)
                height = int(detection[3] * h)
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                cv2.circle(frame, (x, y), 4, (0, 255, 255), 5)
                cv2.putText(frame, "xy...", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 0), 2)

                boxes.append([x, y + height,x + width, y + height])
                confidences.append(float(confidence))
                classIDs.append(classID)
                # draw a bounding box around the detected object and label the confidence of the detections
                # draw a bounding box rectangle and label the confidence of the detections
                cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)
                text = "{}: {:.4f}".format(classes[classID], confidence)
                cv2.putText(frame, text , (x, y + height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                plt.imshow(frame)
                # get rotated bounding box corners
    # loop over the bounding boxes and confidences
    for (box, confidence,classID) in zip((boxes), (confidences), (classIDs)):
        print('boxes:',boxes , 'box:', box)
        # get the angle of the rotated bounding box
        angle = np.arctan2(box[1][0] - box[0][0], box[1][1] - box[0][1])
        # get the rotated bounding box
        x, y, w, h = get_rotated_bbox(box, angle)
        # get the rotated bounding box corners
        corners = get_rotated_bbox_corners(box, angle)
        # draw a bounding box rectangle and label the confidence of the detections
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        text = "{}: {:.4f}".format(classes[classID], confidence)
        cv2.putText(frame, text , (x, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        # draw a line between the center of the rotated bounding box and the center of the original bounding box
        cv2.line(frame, (int(x + w/2), int(y + h/2)), (int(x), int(y)), (255, 0, 0), 2)
        # draw a line between the center of the rotated bounding box and the center of the original bounding box
        cv2.line(frame, (int(x + w/2), int(y + h/2)), (int(x), int(y)), (255, 0, 0), 2)
        # draw a line between the center of the rotated bounding box and the center of the original bounding box
        cv2.line(frame, (int(x + w/2), int(y + h/2)), (int(x), int(y)), (255, 0, 0), 2)
        # draw a line between the center of the rotated bounding box and the center of the original bounding box
        cv2.line(frame, (int(x + w/2), int(y + h/2)),  (int(x), int(y)), (255, 0, 0), 2)
        # draw a line between the center of the original bounding box and the center of the rotated bounding box
        cv2.line(int(x + w/2), int(y + h/2), (int(x), int(y)), (255, 0, 0), 2)

    # show the output frame and wait for a keypress


    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    # if the q key was pressed, break from the loop
    if key == ord("q"):
        break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()