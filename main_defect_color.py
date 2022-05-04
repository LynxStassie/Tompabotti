import optparse
from builtins import print

import cv2
from datetime import datetime

from numpy import dtype

from Functions import*
import matplotlib.pyplot as plt

def now():
    # datetime object containing current date and time
    now = datetime.now()

    # print("now =", now)

    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    # print("date and time =", dt_string)
    return dt_string
# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]

# Draw the predicted bounding box
def drawPred(result_image, classId, conf, left, top, right, bottom):
    if first_pass:
        color = 0, 255, 0
    else:
        color = 255, 255, 255
    cv.rectangle(result_image, (left, top), (right, bottom), color, 3, 3)

def drawBad(result_image, left, top, right, bottom):
    cv.rectangle(result_image, (left, top), (right, bottom), (0, 0, 255), 3, 4)

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(color_image, result_image, outs, confThreshold, nmsThreshold):
    frameHeight = color_image.shape[0]
    frameWidth = color_image.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        # print("out.shape : ", out.shape)
        for detection in out:
            # if detection[4]>0.001:
            scores = detection[5:]
            classId = np.argmax(scores)
            # if scores[classId]>confThreshold:
            confidence = scores[classId]
            # if detection[4] > tconfThreshold:
            #     print(detection[4], " - ", scores[classId], " - th : ", confThreshold)
            if confidence > tconfThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)

                if not first_pass:
                    center_x = center_x + rectangle_of_tomatoes[l][0]
                    center_y = center_y + rectangle_of_tomatoes[l][1]

                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    k = 0
    for i in indices:
        i = i
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(result_image, classIds[i], confidences[i], left, top, left + width, top + height)
        if not first_pass:
            rectangle_of_defects.append([int(left), int(top), int(width), int(height)])
        if first_pass:
            crop_imgs.append(color_image[top:top + height, left:left + width])
            rectangle_of_tomatoes.append([int(left), int(top), int(width), int(height)])
            k = k + 1

path = 'C:/Users/123456/TOMATO'
files = os.listdir(path)

# Start the main loop for the whole system
# for f in files:
cnt_im=0
profile = pipeline.start(config)
while True:
    frames = pipeline.wait_for_frames()
    #intr = rs.video_stream.intrinsics.fx
    #intrfx = profile.get_stream(rs.stream.depth)
    #print("intrinsic fx ", intrfx)
    #intrfy = rs.intrinsics.fy
    #print("intrinsic fy ", intrfy)
    #intr1 = rs.intrinsics.ppx
    #print ("ppx ", intr1)
    #intr2 = rs.intrinsics.ppy
    #print("ppy ", intr2)
    # color sensor
    color_sensor = profile.get_device().query_sensors()[1]
    color_sensor.set_option(rs.option.enable_auto_exposure, 1)
    align = rs.align(rs.stream.color)
    frameset = align.process(frames)

    # advanced mode
    path_to_settings_file = '../med_den.json'#'../hole_f_on.json'
    with open(path_to_settings_file, 'r') as file:
        json_text = file.read().strip()
    device = rs.context().devices[0]
    print(device)
    advanced_mode = rs.rs400_advanced_mode(device)
    advanced_mode.load_json(json_text)
    print("advanced_mode.is_enabled=", advanced_mode.is_enabled())
    print(advanced_mode)
    # Get aligned frames
    depth_frame_filter = frameset.get_depth_frame()



    #color_frame = frames.get_color_frame()
    color_frame = frameset.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    cnt_im+=1
    #====saving image
    # Image directory
    directory = r'D:\tomato_images'
    # List files and directories
    filename =directory + '\\'+'im' + str(now()) + '.jpg'
    # Saving the image
    cv2.imwrite(filename, color_image)
    print(filename)

    # List files and directories
    # in 'C:/Users / Rajnish / Desktop / GeeksforGeeks'
    print("After saving image:")
    print(os.listdir(directory))

    print('Successfully saved')
    #====saving image

    depth_sensor = profile.get_device().query_sensors()[0]
    depth_sensor.set_option(rs.option.enable_auto_exposure, True)

    #exposure_value = depth_sensor.get_option(rs.option.exposure)  # Get exposure
    #gain_value = depth_sensor.get_option(rs.option.gain)  # Get exposure

    depth_frame = frameset.get_depth_frame()


    depth_image = np.asanyarray(depth_frame.get_data())

    colorizer = rs.colorizer()
    colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
    colorized_depthColorMap = cv.applyColorMap(cv.convertScaleAbs(colorized_depth, alpha=1.5), cv2.COLORMAP_JET)

    cv.putText(colorized_depth, str("exposure=" + str("exposure_value")), (50 - 30, 50 - 20), cv.FONT_HERSHEY_SIMPLEX, 1,
               (255, 255, 100), 2)
    cv.putText(colorized_depth, str("gain=" + str("gain_value")), (50 - 30, 100 - 20), cv.FONT_HERSHEY_SIMPLEX, 1,
               (100, 255, 100), 2)

    plt.imshow(color_image)#, alpha=0.6)
    #plt.imshow(depth_image, alpha=0.8)
    plt.imshow(colorized_depth, alpha=0.6)
    plt.show()
    #key = cv.waitKey(1)

    # full_path = path + f
    qualify_image = color_image
    depth_picture = depth_image
    #pipeline.stop()

    qualify_depth = depth_picture.copy()
    qualify_result = qualify_image.copy()

    #show_process_image('Start qualify',qualify_result)

    first_pass = True
    # Create necessary lists
    crop_imgs = [] #crop images for each single tomato
    rectangle_of_tomatoes = [] #coordinate of each tomato
    rectangle_of_defects = [] #coordinate of each defect
    nondefect_tomatoes = [] #list of tomatoes dont have defects on it
    bad_tomatoes = [] #list of bad tomatoes

    # Neural Net
    #############################################################################################
    net = cv.dnn.readNetFromDarknet(tmodelConfiguration, tmodelWeights)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    # timg = cv.imread(timgPath)
    timg = qualify_image

    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(timg, 1 / 255, (tinpWidth, tinpHeight), [0, 0, 0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))

    # Remove the bounding boxes with low confidence
    postprocess(timg, qualify_result, outs, tconfThreshold, tnmsThreshold)


    tomato_coords = coords(rectangle_of_tomatoes, qualify_result, qualify_image, qualify_depth, colorized_depth, profile)
    # Show result with the cutting points
    show_process_image('Qualify result', qualify_result)

    # Send coordinates to Robot to execute cutting command
    # send_bad_cutting_info(bad_tomato_coordinates)

    ################################Weighting Staion###################################################
    # Waiting for robot's signal
    # while True:
    #     if receive_data_to_weight():
    #         break
    #     else:
    #         continue

    result_image = qualify_image #getframe()
    # weight_image = cv.imread(full_path)
    result = result_image.copy()
    show_process_image('Start weight', result)



# Press 'q to stop
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break