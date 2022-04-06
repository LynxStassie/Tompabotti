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
######################################Qualify Staion################################################################
    # Waiting for robot's signal
    # while True:
    #     if receive_data_to_qualify():
    #         break
    #     else:
    #         continue

#=====
    # Create alignment primitive with color as its target stream:
    # see https://github.com/ut-robotics/picr21-team-4meats/blob/0a2f68959e92fb180e8dc32ea1351e628a1b4e30/camera.py#L73
    #
    exp = 71.0
    gain = 8
    exposure_delta = 100
    print("exposure value exp=",exp)
    # initialize camera objects
    profile = pipeline.start(config)
    frames = pipeline.wait_for_frames()
    align = rs.align(rs.stream.color)
    frameset = align.process(frames)
    color_sensor = profile.get_device().query_sensors()[1]
    # color sensor - turn on auto_ exposure and whBalance
    color_sensor.set_option(rs.option.enable_auto_white_balance, True)
    color_sensor.set_option(rs.option.enable_auto_exposure, True)

    # prepare vars to output initial values


    # set delta for exposure increment
    while (1):
        ## filter options
        # color_sensor.set_option(rs.option.exposure, 10.0)
        # color_sensor.set_option(rs.option.enable_auto_exposure,True)
        # color_sensor.set_option(rs.option.hdr_enabled, 1)
        depth_sensor = profile.get_device().query_sensors()[0]
        depth_sensor.set_option(rs.option.enable_auto_exposure, True)
        #exp = depth_sensor.get_option(rs.option.exposure)
        #gain = depth_sensor.get_option(rs.option.gain)
        #print("initial: exposure=",exp, "gain=", gain)
        #man_exposure = depth_sensor.set_option(rs.option.exposure, exp)
        # man_exposure = depth_sensor.set_option(rs.option.exposure, exp)
        #man_gain = depth_sensor.set_option(rs.option.exposure, 8.0)
        #print("man_gain=",man_gain,"man_exp=",man_exposure)

        # exposure_min, exposure_max = depth_sensor.get_option_range(rs.option.exposure).min, ...
        # depth_sensor.get_option_range(rs.option.exposure).max
        # exposure_limit = depth_sensor.get_option_range(rs.option.auto_exposure_limit)
        # print(exposure_limit, exposure_max, exposure_min,exposure_delta)
        # sensor.set_option(rs.option.exposure, 1.0)
        #sensor.set_option(rs.option.gain, 8.0)

        time.sleep(1)

        exposure_value = depth_sensor.get_option(rs.option.exposure)  # Get exposure
        gain_value = depth_sensor.get_option(rs.option.gain)  # Get exposure
        print(exposure_value,gain_value)
        #auto_exp = sensor.get_option(rs.option.enable_auto_exposure)
       # print(exp, auto_exp)
        # explosure = 50.0q
        #print("exposure=", exp, exposure_min, exposure_max)
        # set_short_range(depth_sensor)
        # auto_expl = depth_sensor.get_option(rs.option.enable_auto_exposure)
        # print("auto_exposure=",auto_expl)
        # auto_expl = 2.0
        # print("auto_exposure=",auto_expl)
        # exp=1
       # exposure = exp
        color_frame = frameset.get_color_frame()
        depth_frame = frameset.get_depth_frame()
        depth_frame2=depth_frame
        color = np.asanyarray(color_frame.get_data())
        #plt.rcParams["axes.grid"] = False
        #plt.rcParams['figure.figsize'] = [12, 6]
        #plt.imshow(color)
        #plt.show()

        #plt.imshow(colorized_depth)
        #plt.show()

        colorizer = rs.colorizer()
        colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        colorized_depthColorMap=cv.applyColorMap(cv.convertScaleAbs(colorized_depth, alpha=14), cv.COLORMAP_OCEAN)
        #plt.imshow(colorized_depthColorMap)
        #plt.show()
        # Show the two frames together:
        #images = np.hstack((color, colorized_depth))
        #plt.imshow(images)
        #plt.show()
        # exp = exp + exposure_delta
        # print("exposure=",exp)
        #exposure = depth_sensor.set_option(rs.option.exposure, exp)
        #exposure = depth_sensor.set_option(rs.option.exposure, exp)
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