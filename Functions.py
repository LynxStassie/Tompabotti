import os
import cv2 as cv
import numpy as np
from PIL.ImageOps import crop
from pywinusb import hid
from time import sleep
import time
import pyrealsense2 as rs
import socket
from statistics import median
from math import sqrt
from PIL import Image
from itertools import count
import matplotlib.pyplot as plt


image_name_counter = 0

# Parameters for tomato recognition
tconfThreshold = 0.01  # Confidence threshold
tnmsThreshold = 0.4  # Non-maximum suppression threshold
tinpWidth = 832  # 608     #Width of network's input image
tinpHeight = 832  # 608     #Height of network's input image

# Parameters for defect recognition
dconfThreshold = 0.01  # Confidence threshold
dnmsThreshold = 0.4  # Non-maximum suppression threshold
dinpWidth = 832  # 608     #Width of network's input image
dinpHeight = 832  # 608     #Height of network's input image

# Load names of tom class
tclassesFile = "tobj.names"

tclasses = None
with open(tclassesFile, 'rt') as f:
    tclasses = f.read().rstrip('\n').split('\n')

# Load names of defect class
dclassesFile = "dobj.names"

dclasses = None
with open(dclassesFile, 'rt') as f:
    dclasses = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
dmodelConfiguration = os.path.join(os.path.dirname(__file__),"yolov3_tiny_3l_mod.cfg")
dmodelWeights =os.path.join(os.path.dirname(__file__),"12_yolov3_tiny_3l_mod_best.weights")

# Give the configuration and weight files for the model and load the network using them.
tmodelConfiguration = os.path.join(os.path.dirname(__file__),"yolov3-tom2.cfg")
tmodelWeights = os.path.join(os.path.dirname(__file__),"05_yolov3-tom2_best.weights")

# Initialize the camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

def getframe():
    pipeline.start(config)
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    pipeline.stop()
    return color_image

def get_distance(point,frames):
    pipeline.start(config)
    #frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    depth_image = np.asarray(depth_frame.get_data())
    distance = depth_image[point[1], point[0]]
    pipeline.stop()
    return distance

def color_classify(img, rate):
    tomato_region = []
    for col in img:
        for pixel in col:
            percentage_red = (int(pixel[2]) / (int(pixel[0]) + int(pixel[1]) + int(pixel[2]))) * 100

            percentage_green = (int(pixel[1]) / (int(pixel[0]) + int(pixel[1]) + int(pixel[2]))) * 100

            if percentage_red > percentage_green and percentage_red > 40 and int(pixel[2]) > 50:
                tomato_region.append(pixel)

    b = 0
    g = 0
    r = 0
    for tomato_pixel in tomato_region:
        try:
            b += int(tomato_pixel[0])
            g += int(tomato_pixel[1])
            r += int(tomato_pixel[2])
        except:
            continue

    red = r / (b + g + r)
    # Show the classifying tomato and print the rate
    show_process_image('Color classify',img)
    print('color rate:',red)

    if red >= rate:
        return True
    else:
        return False

def tomato_detect(img, result_img):

    # Get the names of the output layers
    def getOutputsNames(net):
        # Get the names of all the layers in the network
        layersNames = net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]

    def drawPred(result_img, left, top, right, bottom):
        color = 0, 255, 0
        cv.rectangle(result_img, (left, top), (right, bottom), color, 3, 3)

    def postprocess(img, result_img, outs, confThreshold, nmsThreshold):
        frameHeight = img.shape[0]
        frameWidth = img.shape[1]

        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []
        for out in outs:
            # ("out.shape : ", out.shape)
            for detection in out:
                confidence = detection[5]
                if confidence > tconfThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(0)
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
            drawPred(result_img, left, top, left + width, top + height)
            # Crop found tomatoes
            # crop_imgs.append(img[top:top + height, left:left + width])
            tomato_boxes.append([int(left), int(top), int(width), int(height)])
            k = k + 1

    tomato_boxes = []
    net = cv.dnn.readNetFromDarknet(tmodelConfiguration, tmodelWeights)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(img, 1 / 255, (tinpWidth, tinpHeight), [0, 0, 0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))

    postprocess(img,result_img, outs, tconfThreshold, tnmsThreshold)

    return tomato_boxes

def show_process_image(window_name, image):
    cv.destroyAllWindows()
    cv.imshow(window_name,image)
    im = Image.fromarray(image)
    global image_name_counter
    image_name_counter = image_name_counter + 1
    # target dir - dirpath that in dirname
    # C:\Users\123456\Tompabotti\Tomato_project\Results_photo
    dirpath=os.path.join(os.path.dirname(__file__))
    dirname='Results_photo'
    filename='tomato.jpeg'
    im.save(os.path.join(os.path.join(dirpath, dirname ).replace('Program\\',''), str(image_name_counter) + filename ))
    cv.waitKey(1)

#####################################WEIGHT#####################################
def draw_overweight(image,tomato_boxes,overweight_boxes):
    if overweight_boxes == None:
        pass

    else:
        for tomato_box in tomato_boxes:
            left = tomato_box[0]
            top = tomato_box[1]
            right = tomato_box[0]+tomato_box[2]
            bottom = tomato_box[1]+tomato_box[3]
            cv.rectangle(image, (left, top), (right, bottom), (0,255,0), 3, 3)

        for overweight_box in overweight_boxes:
            left = overweight_box[0]
            top = overweight_box[1]
            right = overweight_box[0] + overweight_box[2]
            bottom = overweight_box[1] + overweight_box[3]
            cv.rectangle(image, (left, top), (right, bottom), (255, 255, 255), 3, 3)

def sample_handler(data):
    if data[2] == 2:
        weight = data[4] + data[5] * 256
    if data[2] == 11:
        ouncesFactor = 10 ** data[3] - 256
        weight = data[4] + (data[5] * 256) * ouncesFactor

    # Write weight to external file
    weight_file = open('C:/Users/123456/Tompabotti/Tomato_project/weight.txt','w')
    weight_file.write(str(weight))
    weight_file.close()

def weight():
    # vendorID = 0x0922
    # productID = 0x8003
    # device = hid.HidDeviceFilter(vendor_id = vendorID, product_id = productID).get_devices()[0]
    #
    # device.open()
    # set custom raw data handler
    # device.set_raw_data_handler(sample_handler)
    # sleep(0.3)
    #
    # device.close()

    # Read weight file
    #weight_file = open(os.path.join(os.path.dirname(__file__),'weight.txt'), 'r')
    weight = 902#int(weight_file.read())
    #weight_file.close()
    return weight

def weight_evaluate(weight, number_of_tomatoes):
    if weight < 495:
        number_cut = None
    else:
        try:
            avg_tom_weight = weight / number_of_tomatoes
            number_cut = int((weight - 490) / avg_tom_weight)
        except:
            print('Error in weight_evaluate')

    return number_cut

def determine_overweight(tomato_boxes, num_cut):
    if num_cut == None:
        overweight = None
        print('Not enough weight')

    elif num_cut == 0:
        overweight = []
        print('No cut needed')

    else:
        tomato_boxes.sort(key = lambda x: x[0])
        overweight = tomato_boxes[:num_cut]

    return overweight

#####################################PEDICELDETECT#####################################
def pedicel_detect(orig):

    def get_output_layers(net):
        layer_names = net.getLayerNames()

        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

        return output_layers


    def draw_prediction(result_image, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = str(classes[class_id])

        color = COLORS[class_id]

        cv.rectangle(result_image, (x, y), (x_plus_w, y_plus_h), color, 2)

        cv.putText(result_image, label, (x - 10, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    
    config = os.path.join(os.path.dirname(__file__),"yolov3_pedicel.cfg")
    weights = os.path.join(os.path.dirname(__file__),"yolov3_pedicel2.weights")
    classes1 = os.path.join(os.path.dirname(__file__),"yolo_pedicel.txt")
    detect_times = 0
    while detect_times <= 5:
        if detect_times == 0:
            image = orig
        if detect_times == 1:
            image = cv.rotate(orig, cv.ROTATE_90_CLOCKWISE)
        if detect_times == 2:
            image = cv.rotate(orig, cv.ROTATE_90_COUNTERCLOCKWISE)
        if detect_times == 3:
            image = cv.rotate(orig, cv.ROTATE_180)
        if detect_times == 4:
            image = cv.flip(orig, 0)
        if detect_times == 5:
            image = cv.flip(orig, 1)

        result_image = image.copy()

        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392

        classes = None

        with open(classes1, 'r') as f:
            classes = [line.strip() for line in f.readlines()]

        COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

        net = cv.dnn.readNet(weights, config)

        blob = cv.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)

        outs = net.forward(get_output_layers(net))

        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        # Thực hiện xác định bằng HOG và SVM
        start = time.time()

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        result = []
        for i in indices:
            i = i
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            draw_prediction(result_image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))
            result.append([[round(x),round(y),round(w),round(h)], confidences[i]])

        #Only show if there is object detected
        if boxes:
            show_process_image("Pedicel detection", result_image)

        # Take only the 1-object-result
        if len(result) != 1:
            pass
        else:
            break

        detect_times += 1

    try:
        return [result[0][0],result_image[0][1],detect_times]
    except:
        return None

def pedicel_coor_convert(tomato, pedicel_box, convert_angle):
    # Center of pedicel box
    m = int(pedicel_box[0] + pedicel_box[2] / 2)
    n = int(pedicel_box[1] + pedicel_box[3] / 2)

    if convert_angle == 0:  # Non-change
        i = int(m + tomato[0])
        j = int(n + tomato[1])
    if convert_angle == 1:  # 90-CW
        i = int(n + tomato[0])
        j = int((tomato[2] - m) + tomato[1])
    if convert_angle == 2:  # 90-CCW
        i = int((tomato[3] - n) + tomato[0])
        j = int(m + tomato[1])
    if convert_angle == 3:  # 180
        i = int((tomato[3] - m) + tomato[0])
        j = int((tomato[2] - n) + tomato[1])
    if convert_angle == 4:  # flip vertical
        i = int(m + tomato[0])
        j = int((tomato[2] - n) + tomato[1])
    if convert_angle == 5:  # flip horizontal
        i = int((tomato[3] - m) + tomato[0])
        j = int(n + tomato[1])

    return (i,j)

def xyz_converter(xframe,yframe,depth):
    # factor = (depth-287)*0.0233/20 + 0.2828
    # factor = 0.2828
    # factor290 = 0.1943
    ref_point = (-644.7,-154.8)
    b_point = (-639.6,221.9)
    c_point = (-437.1, -156.3)

    ab = sqrt((b_point[0] - ref_point[0]) ** 2 + (b_point[1] - ref_point[1]) ** 2)
    ac = sqrt((c_point[0] - ref_point[0]) ** 2 + (c_point[1] - ref_point[1]) ** 2)
    factor = (ab/1280 + ac/720)/2

    xUR = ref_point[0] + yframe * factor
    yUR = ref_point[1] + xframe * factor

    # d_cam_station = 350
    # delta_station_robot = -22
    # zUR =  d_cam_station - depth + delta_station_robot

    zUR = 73 - depth
    return (xUR,yUR,zUR)

def pedicel_info_process(tomato_boxes, color_image, result_image):
    if tomato_boxes == None:
        pedicel_tomato_coordinates = None

    else:
        common_depth = 277 #mm
        pedicel_tomato_coordinates = []

        for tomato_box in tomato_boxes:
            # Crop the images to detect pedicel in the crop
            crop_img = color_image[tomato_box[1]:tomato_box[1] + tomato_box[2],
                           tomato_box[0]:tomato_box[0] + tomato_box[3]]

            #Center of the tomato bounding box
            (m,n) = (int(tomato_box[0]+tomato_box[2]/2), int(tomato_box[1]+tomato_box[3]/2))
            cv.circle(result_image, (m, n), 4, (0, 255, 255), 5)

            # pedicel = [pedicel_box, confidence, angle_detected]
            pedicel = pedicel_detect(crop_img)
            if pedicel == None:
                print('cannot find pedicel')
                pass
            else:
                (i, j) = pedicel_coor_convert(tomato_box, pedicel[0], pedicel[2])
                print('pedicel detected')
                cv.circle(result_image, (i, j), 4, (0, 0, 255), 5)

                # Find coordinate of the point
                depth_avg = 0
                detect_depth_time = 0
                depth_list = []
                while True:
                    depth = get_distance((i,j))
                    # depth = 277
                    depth = 1
                    print('depth:', depth)
                    if depth != 0:
                        depth_list.append(depth)
                    detect_depth_time += 1

                    if len(depth_list) >= 5 or detect_depth_time >=10:
                        break

                try:
                    depth_avg = median(depth_list)
                except:
                    depth_avg = common_depth

                # depth_avg = common_depth #Use default depth

                cut_coordinate = xyz_converter(i,j,depth_avg)
                tomato_coordinate = xyz_converter(m,n,depth_avg+10)

                try:
                    pedicel_tomato_coordinates.append([cut_coordinate,tomato_coordinate])
                except:
                    print('coordinate error')
                    break

                # Print out coordinates
                print('x:', cut_coordinate[0])
                print('y:', cut_coordinate[1])
                print('z:', cut_coordinate[2])
                x_label = 'x:%.d' % cut_coordinate[0]
                y_label = 'y:%.d' % cut_coordinate[1]
                z_label = 'z:%.d' % cut_coordinate[2]
                cv.putText(result_image, x_label, (i - 30, j - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv.putText(result_image, y_label, (i - 30, j), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv.putText(result_image, z_label, (i - 30, j + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return pedicel_tomato_coordinates


def coords(tomato_boxes, color_image, result_image, depth_image): #,depth_frame, frameset,aligned_depth_frame):
    # находит координаты центра квадрата из tomato_boxes
    # координаты x , y - это координаты пикселя в матрице, это координаты центра бокса из
    if tomato_boxes == None:
        pedicel_tomato_coordinates = None

    else:
        common_depth = 50 #mm
        pedicel_tomato_coordinates = []

        for tomato_box in tomato_boxes:
            # Crop the images to detect pedicel in the crop

            xmin = tomato_box[0]
            ymin = tomato_box[1]
            xmax = tomato_box[0]+tomato_box[2] # xmin + width
            ymax = tomato_box[1] + tomato_box[3] # ymin + height


            (m,n) = (int((xmin+xmax)/2), int((ymin+ymax)/2)) # toouple of coords of center of the box with tomato
            cv.circle(result_image, (m, n), 4, (0, 255, 255), 5)

            (i, j) = (m,n)#pedicel_coor_convert(tomato_box, pedicel[0], pedicel[2])
            print('coordinate detected')


            # Print out coordinates
            x_coord = (m,n)[0]
            y_coord = (m,n)[1]
            distance = getDist2(x_coord,y_coord,depth_image,color_image)
            #distance = aligned_depth_frame[x_coord,y_coord].astype(float)
            print('x:', x_coord)#cut_coordinate[0])
            print('y:', y_coord)#cut_coordinate[1])
            print('z:', distance)
            x_label = 'x:%.d' % x_coord
            y_label = 'y:%.d' % y_coord
            z_label = 'z:%.3f' % distance
            cv.circle(result_image, (i, j), 4, (0, 0, 255), 5)
            cv.putText(result_image, x_label, (i - 30, j - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv.putText(result_image, y_label, (i - 30, j), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv.putText(result_image, z_label, (i - 30, j + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            plt.imshow(result_image)
            plt.show()
  #  (i - 30, j + 20)
    return pedicel_tomato_coordinates


def getDist2(m,n,depth_img,color_img):
    # frame_width = 1280
    # frame_heigth = 720
    # pipeline2=rs.pipeline()
    # config.enable_stream(rs.stream.depth, frame_width, frame_heigth, rs.format.z16, 30)
    # config.enable_stream(rs.stream.color, frame_width, frame_heigth, rs.format.bgr8, 30)
    #
    # profile = pipeline2.start(config)
    # # Getting the depth sensor's depth scale (see rs-align example for explanation)
    # # depth_sensor = profile.get_device().first_depth_sensor()
    # # depth_scale = depth_sensor.get_depth_scale()
    # # print("Depth Scale is: " , depth_scale)
    #
    # depth_sensor = profile.get_device().first_depth_sensor()
    # auto_expl = depth_sensor.get_option(rs.option.enable_auto_exposure)
    # print("ae=",auto_expl)
    # auto_expl = 2.0
    # laser_pwr = depth_sensor.get_option(rs.option.laser_power)
    # print("laser power = ", laser_pwr)
    # # print(depth_sensor.get_option_range(rs.option.min_distance))
    # # print("laser power range = " , laser_range.min , "~", laser_range.max)
    # set_laser = 8
    # # depth_sensor.set_option(rs.option.min_distance, 0.25)
    # depth_scale = depth_sensor.get_depth_scale()
    # print(depth_sensor.get_depth_scale())
    # title = 'mouse event'
    #
    #
    # # Wait for a coherent pair of frames: depth and color
    # frames = pipeline2.wait_for_frames()
    # depth_frame = frames.get_depth_frame()
    # color_frame = frames.get_color_frame()
    #
    #
    # # Convert images to numpy arrays
    # depth_image = np.asanyarray(depth_frame.get_data())
    # color_image = np.asanyarray(color_frame.get_data())
    # # depth_image_3d = np.dstack((depth_image,depth_image,depth_image))
    # # print(np.shape(depth_image))
    # # print(np.shape(color_image))
    # depth_colormap = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=4), cv.COLORMAP_JET)
    # # depth_colormap_3d = np.dstack((depth_colormap,depth_colormap,depth_colormap))
    # # depth_colormap = cv2.cvtColor(np.float32(depth_colormap))
    # # cv2.imshow(title, color_frame)
    # # cv2.imshow(title,color_image)
    cv.circle(color_img, (m, n), 4, (255, 255, 255), 5)
    cv.circle(depth_img, (m, n), 4, (255, 255, 255), 5)
    cv.imshow("title", color_img)

    cv.imshow("title", depth_img)


    depth = depth_img[m, n].astype(float)
    return depth



def getDist(xmin, ymin,xmax, ymax):
    # Setup:
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    profile = pipe.start(cfg)

    # Skip 5 first frames to give the Auto-Exposure time to adjust
    for x in range(5):
        pipe.wait_for_frames()

    # Store next frameset for later processing:
    frameset = pipe.wait_for_frames()
    color_frame = frameset.get_color_frame()
    depth_frame = frameset.get_depth_frame()

    # Cleanup:
    pipe.stop()
    #print("Frames Captured")

    color = np.asanyarray(color_frame.get_data())
    plt.rcParams["axes.grid"] = False
    plt.rcParams['figure.figsize'] = [12, 6]
    plt.imshow(color)
    # plt.show()
    colorizer = rs.colorizer()
    colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
    plt.imshow(colorized_depth)
    # plt.show()

    # Create alignment primitive with color as its target stream:
    align = rs.align(rs.stream.color)
    frameset = align.process(frameset)

    # Update color and depth frames:
    aligned_depth_frame = frameset.get_depth_frame()
    colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())

    xmin_depth = xmin
    ymin_depth = ymin
    xmax_depth = xmax
    ymax_depth = ymax
    xmin_depth, ymin_depth, xmax_depth, ymax_depth
    cv.rectangle(colorized_depth, (xmin_depth, ymin_depth),
                 (xmax_depth, ymax_depth), (255, 255, 255), 2)
    plt.imshow(colorized_depth)
    plt.show()

    depth = np.asanyarray(aligned_depth_frame.get_data())
    # Crop depth data:
    #depth = depth[xmin_depth:xmax_depth,ymin_depth:ymax_depth].astype(float)
    depth_colormap = cv.applyColorMap(cv.convertScaleAbs(depth, alpha=1), cv.COLORMAP_JET)
    # Get data scale from the device and convert to meters
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    print(depth_scale)
    depth = depth_colormap * depth_scale
    dist,_,_,_ = cv.mean(depth)

    print("Detected a {0} {1:.3} meters away. and xmin={2} xmin_depth={3} xmax={4} xmin={5} xmax_depth={6}".format("className", dist, xmin,xmin_depth,xmax,xmax_depth,1))

    return float(dist)

def coords2(tomato_boxes, color_image, result_image, profile, outs, aligned_depth_frame, frameset): #,depth_frame, frameset,aligned_depth_frame):
    # Setup:
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    profile = pipe.start(cfg)

    # Skip 5 first frames to give the Auto-Exposure time to adjust
    for x in range(5):
        pipe.wait_for_frames()

    # Store next frameset for later processing:
    frameset = pipe.wait_for_frames()
    color_frame = frameset.get_color_frame()
    depth_frame = frameset.get_depth_frame()

    # Cleanup:
    pipe.stop()
    print("Frames Captured")

    color = np.asanyarray(color_frame.get_data())
    plt.rcParams["axes.grid"] = False
    plt.rcParams['figure.figsize'] = [12, 6]
    plt.imshow(color)
    # plt.show()
    colorizer = rs.colorizer()
    colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
    plt.imshow(colorized_depth)
    # plt.show()

    # Create alignment primitive with color as its target stream:
    align = rs.align(rs.stream.color)
    frameset = align.process(frameset)

    # Update color and depth frames:
    aligned_depth_frame = frameset.get_depth_frame()
    colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())

    # Show the two frames together:
    images = np.hstack((color, colorized_depth))
    plt.imshow(images)
    # plt.show()

    # Standard OpenCV boilerplate for running the net:
    height, width = color.shape[:2]
    expected = 300
    aspect = width / height
    resized_image = cv.resize(color, (round(expected * aspect), expected))
    crop_start = round(expected * (aspect - 1) / 2)
    crop_img = resized_image[0:expected, crop_start:crop_start + expected]




    if tomato_boxes == None:
        pedicel_tomato_coordinates = None

    else:
        common_depth = 50 #mm
        pedicel_tomato_coordinates = []

        for tomato_box in tomato_boxes:
            # Crop the images to detect pedicel in the crop
            crop_img = color_image[tomato_box[1]:tomato_box[1] + tomato_box[2],
                           tomato_box[0]:tomato_box[0] + tomato_box[3]]

            #Center of the tomato bounding box
            (m,n) = (int(tomato_box[0]+tomato_box[2]/2), int(tomato_box[1]+tomato_box[3]/2))
            cv.circle(result_image, (m, n), 4, (0, 255, 255), 5)

            # label = detections[0, 0, 0, 1]
            # conf = detections[0, 0, 0, 2]
            xmin = tomato_box[0]
            ymin = tomato_box[1]
            xmax = tomato_box[2]
            ymax = tomato_box[3]

            # className = classNames[int(label)]
            #print(crop_img)
            cv.rectangle(crop_img, (int(xmin * expected), int(ymin * expected)),
                         (int(xmax * expected), int(ymax * expected)), (255, 255, 255), 2)
            cv.putText(crop_img, "surprise!",
                       (int(xmin * expected), int(ymin * expected) - 5),
                       cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))

            show_process_image('Start qualify', crop_img)
            plt.imshow(crop_img)
            plt.show()
            scale = height / expected /1000
            xmin_depth = int((xmin * expected + crop_start) * scale)
            ymin_depth = int((ymin * expected) * scale)
            xmax_depth = int((xmax * expected + crop_start) * scale)
            ymax_depth = int((ymax * expected) * scale)
            xmin_depth, ymin_depth, xmax_depth, ymax_depth
            cv.rectangle(colorized_depth, (xmin_depth, ymin_depth),
                         (xmax_depth, ymax_depth), (255, 255, 255), 2)
            plt.imshow(colorized_depth)
            # plt.show()

            depth = np.asanyarray(aligned_depth_frame.get_data())
            # Crop depth data:
            depth = depth[xmin_depth:xmax_depth, ymin_depth:ymax_depth].astype(float)

            # Get data scale from the device and convert to meters
            depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            depth = depth * depth_scale
            dist, _, _, _ = cv.mean(depth)

            print(
                "Detected a {0} {1:.3} meters away. and xmin={2} xmin_depth={3} xmax={4} xmin={5} xmax_depth={6}".format(
                    "surprise!", dist, xmin, xmin_depth, xmax, xmax_depth, 1))

            # pedicel = [pedicel_box, confidence, angle_detected]
            pedicel = pedicel_detect(crop_img)
            if pedicel == None:
                print('cannot find pedicel')
                pass
            else:
                (i, j) = pedicel_coor_convert(tomato_box, pedicel[0], pedicel[2])
                print('pedicel detected')
                cv.circle(result_image, (i, j), 4, (0, 0, 255), 5)

                # Find coordinate of the point
                depth_avg = 0
                detect_depth_time = 0
                depth_list = []
                while True:
                    depth = get_distance((i,j))
                    # depth = 277
                    depth = 1
                    print('depth:', depth)
                    if depth != 0:
                        depth_list.append(depth)
                    detect_depth_time += 1

                    if len(depth_list) >= 5 or detect_depth_time >=10:
                        break

                try:
                    depth_avg = median(depth_list)
                except:
                    depth_avg = common_depth

                # depth_avg = common_depth #Use default depth

                cut_coordinate = xyz_converter(i,j,depth_avg)
                tomato_coordinate = xyz_converter(m,n,depth_avg+10)

                try:
                    pedicel_tomato_coordinates.append([cut_coordinate,tomato_coordinate])
                except:
                    print('coordinate error')
                    break




    try:
        pedicel_tomato_coordinates.append([cut_coordinate,tomato_coordinate])
    except:
        print('coordinate error')
    return pedicel_tomato_coordinates

#####################################COMMUNICATE#####################################
def send_bad_tomato_amount(a):
    HOST = "192.168.0.45"  # The remote host
    PORT = 30002  # The same port as used by the server
    print("Wait to send bad tomato amount...")

    while True:

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))  # Bind to the port
        s.listen(2)  # Now wait for client connection.
        c, addr = s.accept()  # Establish connection with client.

        try:
            msg = str(c.recv(1024))
            time.sleep(0.5)
            if msg == "b'asking_for_bad_amount'":
                print(msg)
                time.sleep(0.5)
                position = (f'"({a})"')
                final_val = (position.encode('ASCII'))
                c.send(final_val)
                break

        except socket.error as socketerror:
            print('Send amount error')


    c.close()
    s.close()

def send_not_enough_weight():
    HOST = "192.168.0.45"  # The remote host
    PORT = 30002  # The same port as used by the server
    print("Wait to send remove tomato...")

    while True:

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))  # Bind to the port
        s.listen(2)  # Now wait for client connection.
        c, addr = s.accept()  # Establish connection with client.

        try:
            msg = str(c.recv(1024))
            time.sleep(0.5)
            if msg == "b'asking_for_weight_amount'":
                print(msg)
                time.sleep(0.5)
                position = (f'"({-1})"')
                final_val = (position.encode('ASCII'))
                c.send(final_val)
                break

        except socket.error as socketerror:
            print('Send amount error')


    c.close()
    s.close()

def send_overweight_tomato_amount(b):
    HOST = "192.168.0.45"  # The remote host
    PORT = 30002  # The same port as used by the server
    print("Wait to send overweight tomato amount...")

    while True:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))  # Bind to the port
        s.listen(2)  # Now wait for client connection.
        c, addr = s.accept()  # Establish connection with client.

        try:
            msg = str(c.recv(1024))
            time.sleep(0.5)
            if msg == "b'asking_for_weight_amount'":
                print(msg)
                time.sleep(0.5)
                position = (f'"({b})"')
                final_val = (position.encode('ASCII'))
                c.send(final_val)
                break

        except socket.error as socketerror:
            print('Send amount error')

    c.close()
    s.close()

def send_coordinate(x,y,z):

    HOST = "192.168.0.45"  # The remote host
    PORT = 30002  # The same port as used by the server
    print("Sending coordinate...")


    while True:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))  # Bind to the port
        s.listen(2)  # Now wait for client connection.
        c, addr = s.accept()  # Establish connection with client.

        try:
            msg = str(c.recv(1024))
            time.sleep(0.5)
            if msg == "b'asking_for_coordinate'":
                print(msg)
                time.sleep(0.5)
                position = (f'"({x},{y},{z})"')
                final_val = (position.encode('ASCII'))
                c.send(final_val)
                break


        except socket.error as socketerror:
            print('Send coordinate error')


    c.close()
    s.close()

    print("Coordinate sent")

def send_suction_coordinate(x,y,z):

    HOST = "192.168.0.45"  # The remote host
    PORT = 30002  # The same port as used by the server
    print("Sending suction coordinate...")


    while True:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))  # Bind to the port
        s.listen(2)  # Now wait for client connection.
        c, addr = s.accept()  # Establish connection with client.

        try:
            msg = str(c.recv(1024))
            time.sleep(0.5)
            if msg == "b'asking_for_suction_coordinate'":
                print(msg)
                time.sleep(0.5)
                position = (f'"({x},{y},{z})"')
                final_val = (position.encode('ASCII'))
                c.send(final_val)
                break


        except socket.error as socketerror:
            print('Send coordinate error')


    c.close()
    s.close()

    print("Suction coordinate sent")

def receive_data_to_qualify():
    start = bool
    HOST = "192.168.0.45" # The remote host
    PORT = 30002  # The same port as used by the server
    print("Waiting for qualify signal...")

    while True:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))  # Bind to the port
        s.listen(2)  # Now wait for client connection.
        c, addr = s.accept()  # Establish connection with client.
        print('connect success')

        try:
            msg = str(c.recv(1024))
            time.sleep(0.5)
            start = False
            if msg == "b'start_qualify'":
                print(msg)
                time.sleep(0.5)
                start = True
                break

        except socket.error as socketerror:
            print('Trigger data error')

    c.close()
    s.close()
    return start

def receive_data_to_weight():
    start = bool
    HOST = "192.168.0.45"  # The remote host
    PORT = 30002  # The same port as used by the server
    print("Waiting for weight signal...")

    while True:

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))  # Bind to the port
        s.listen(2)  # Now wait for client connection.
        c, addr = s.accept()  # Establish connection with client.

        try:
            msg = str(c.recv(1024))
            time.sleep(1)
            start = False
            if msg == "b'start_weight'":
                print(msg)
                time.sleep(0.5)
                start = True
                break

        except socket.error as socketerror:
            print('Trigger data error')

    c.close()
    s.close()
    return start

def send_bad_cutting_info(cut_tomato_coordinates):

    if len(cut_tomato_coordinates) != 0:

        send_bad_tomato_amount(len(cut_tomato_coordinates))

        for cut_tomato_coordinate in cut_tomato_coordinates:
            send_coordinate(cut_tomato_coordinate[0][0],cut_tomato_coordinate[0][1],cut_tomato_coordinate[0][2]) #send coors of pedicels
            time.sleep(2)
            send_suction_coordinate(cut_tomato_coordinate[1][0],cut_tomato_coordinate[1][1],cut_tomato_coordinate[1][2]) #send coors of tomatoes

    else:
        # Send 0 as tomato amount
        send_bad_tomato_amount(len(cut_tomato_coordinates))

def send_overweight_cutting_info(cut_tomato_coordinates):
    if cut_tomato_coordinates == None:
        send_not_enough_weight()

    elif len(cut_tomato_coordinates) != 0:

        send_overweight_tomato_amount(len(cut_tomato_coordinates))

        for cut_tomato_coordinate in cut_tomato_coordinates:
            send_coordinate(cut_tomato_coordinate[0][0], cut_tomato_coordinate[0][1], cut_tomato_coordinate[0][2]) #send coors of pedicels
            time.sleep(2)
            send_suction_coordinate(cut_tomato_coordinate[1][0], cut_tomato_coordinate[1][1], cut_tomato_coordinate[1][2])  # send coors of tomatoes

    else:
        # Send 0 as tomato amount
        send_overweight_tomato_amount(len(cut_tomato_coordinates))