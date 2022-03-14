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

path = 'C:/Users/123456/pomidor/outpic/'
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
    profile = pipeline.start(config)
    frames = pipeline.wait_for_frames()
#=====

    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    depth_frame2=depth_frame
    color = np.asanyarray(color_frame.get_data())
    plt.rcParams["axes.grid"] = False
    plt.rcParams['figure.figsize'] = [12, 6]
    #plt.imshow(color)
    #plt.show()
    # colorizer = rs.colorizer()
    # colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
    #plt.imshow(colorized_depth)
    #plt.show()

    colorizer = rs.colorizer()
    colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())

    colorized_depthColorMap=cv.applyColorMap(cv.convertScaleAbs(colorized_depth, alpha=14), cv.COLORMAP_OCEAN)
    #plt.imshow(colorized_depthColorMap)
    #plt.show()

    # Create alignment primitive with color as its target stream:
    align = rs.align(rs.stream.color)
    frameset = align.process(frames)

    # Show the two frames together:
    images = np.hstack((color, colorized_depth))
    plt.imshow(images)
    plt.show()
#=====


    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_picture=depth_image
    pipeline.stop()
    qualify_image = color_image

    # full_path = path + f
    # qualify_image = cv.imread(full_path)
    qualify_depth = depth_picture.copy()
    qualify_result = qualify_image.copy()

    show_process_image('Start qualify',qualify_result)

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


    bad_tomato_coordinates = coords(rectangle_of_tomatoes, qualify_result ,qualify_image,qualify_depth,profile)


    # print(bad_tomato_coordinates)
    # print(bad_tomato_coordinates)s

    if bad_tomato_coordinates == []:
        print('Cannot find any pedicel to cut')

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

    weight_image = getframe()
    # weight_image = cv.imread(full_path)
    weight_result = weight_image.copy()
    show_process_image('Start weight', weight_result)

    # Detect tomato
    tomato_boxes = tomato_detect(weight_image,weight_result)
    print(len(tomato_boxes),'tomatoes detected')

    # Read weight value
    #weight = weight()
    weight = 400
    print('weight:',weight)

    # Weight evaluate
    number_cut = weight_evaluate(weight, len(tomato_boxes))
    print('Cut',number_cut,'tomato(es)')

    # Determine the tomatoes needed to cut
    overweight_tomatoes = determine_overweight(tomato_boxes, number_cut)
    # Draw red bb to these tomato
    draw_overweight(weight_result,tomato_boxes,overweight_tomatoes)
    show_process_image('Weight result',weight_result)

    # Detect the pedicels
    overweight_tomato_coordinates = pedicel_info_process(overweight_tomatoes, weight_image, weight_result)

    # Show weight result
    print(overweight_tomato_coordinates)
    show_process_image('Weight result', weight_result)
    plt.imshow(weight_result)
    plt.show()
    # Send coordinates
    #send_overweight_cutting_info(overweight_tomato_coordinates)



    # Calculate depth



# Press 'q to stop
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break