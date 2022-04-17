import optparse

from Functions import*
import matplotlib.pyplot as plt

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
while True:
    # initialize camera objects
    profile = pipeline.start(config)
    frames = pipeline.wait_for_frames()
    # advanced mode
    path_to_settings_file='../default.json'
    with open(path_to_settings_file, 'r') as file:
        json_text = file.read().strip()
    device = rs.context().devices[0]
    print(device)
    advanced_mode = rs.rs400_advanced_mode(device)
    advanced_mode.load_json(json_text)
    print("advanced_mode.is_enabled=",advanced_mode.is_enabled())
    print(advanced_mode)
    # print('ae_control:',advanced_mode.get_ae_control())
    # color sensor
    color_sensor = profile.get_device().query_sensors()[1]
    color_sensor.set_option(rs.option.enable_auto_exposure, 1)
    color_sensor.set_option(rs.option.enable_auto_exposure, 1)
    #color_sensor = profile.get_device().first_color_sensor()

    #roi
    roi_sensor = profile.get_device().first_roi_sensor()

    roi = roi_sensor.get_region_of_interest()
    print(roi.min_x,roi.min_y,roi.max_x,roi.max_y)

    roi = rs.region_of_interest()
    roi.min_x = 600
    roi.min_y = 0
    roi.max_x = 800
    roi.max_y = 100

    # roi.min_x = 100
    # roi.min_y = 100
    # roi.max_x = 200
    # roi.max_y = 200
    roi_sensor.set_region_of_interest(roi)
    color_sensor.set_option(rs.option.enable_auto_exposure, 1)


    roi = roi_sensor.get_region_of_interest()
    print("roi values:",roi.min_x, roi.min_y, roi.max_x, roi.max_y)
    # trying to getting metadata
    # frameset.get_depth_frame().get_frame_metadata(rs2_frame_metadata_value::RS2_FRAME_METADATA_ACTUAL_EXPOSURE) << std::endl;
    actualFrameMetadata = frames.get_depth_frame().get_frame_metadata(rs.frame_metadata_value.actual_exposure)
    print("actual_depth_exp=",actualFrameMetadata)
    print(color_sensor.get_supported_options())
    # color sensor - turn on auto_ exposure and whBalance
    color_sensor.set_option(rs.option.enable_auto_exposure, 1)
    #color_sensor.set_option(rs.option.auto_exposure_priority,0.0)
    sensorState = color_sensor.get_option(rs.option.enable_auto_exposure)
    print('auto_exposure_color=',sensorState)
    #color_sensor.set_option(rs.option.auto_exposure_priority, 1)
    #ae_pr = color_sensor.get_option(rs.option.auto_exposure_priority)
    #print(ae_pr)
    depth_sensor = profile.get_device().query_sensors()[0]
    depth_sensor.set_option(rs.option.enable_auto_exposure, True)

    time.sleep(1)

    exposure_value = depth_sensor.get_option(rs.option.exposure)  # Get exposure


    depth_sensor.set_option(rs.option.emitter_enabled,False)  # Set laser emmtter off

    emitter_enbld_value = depth_sensor.get_option(rs.option.emitter_enabled)  # Get laser
    print('laser enabled', emitter_enbld_value)

    gain_value = depth_sensor.get_option(rs.option.gain)  # Get exposure
    print('exposure_value=',exposure_value,'gain_value=',gain_value)
    align = rs.align(rs.stream.color)
    frameset = align.process(frames)
    color_frame = frameset.get_color_frame()
    # color_frame = frames.get_color_frame()

    depth_frame = frameset.get_depth_frame()
    depth_frame2 = depth_frame
    color = np.asanyarray(color_frame.get_data())

    # color = np.asanyarray(color_sensor.get_data())

    colorizer = rs.colorizer()
    colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
    colorized_depthColorMap=cv.applyColorMap(cv.convertScaleAbs(colorized_depth, alpha=14), cv.COLORMAP_OCEAN)

    depth_image = np.asanyarray(depth_frame.get_data())
    cv.putText(colorized_depth, str("exposure="+str(exposure_value)), (50 - 30, 50 - 20), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 100), 2)
    cv.putText(colorized_depth, str("gain="+str(gain_value)), (50 - 30, 100 - 20), cv.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 100), 2)

    plt.imshow(color)#, alpha=0.6)
    plt.imshow(colorized_depth, alpha=0.9)
    plt.show()
        # break
    color_image = np.asanyarray(color_frame.get_data())
    qualify_image = color_image
    depth_picture=depth_image
    pipeline.stop()

    # full_path = path + f
    # qualify_image = cv.imread(full_path)
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

    result_image = getframe()
    # weight_image = cv.imread(full_path)
    result = result_image.copy()
    show_process_image('Start weight', result)

# Press 'q to stop
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break