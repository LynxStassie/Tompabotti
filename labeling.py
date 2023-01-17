###
# get r-yolo format x_center, y_center, width, height, rotation Theta angle from json min format exported from label studio 

# read json file and return a json dictionary
import json
from math import pi, radians, degrees


# read json file and return it as dictionary
import os.path


def read_json_file(file_name):
    with open(file_name, 'r') as f:
        data = json.load(f)
    return data





if __name__ == "__main__":
    # JSON file to read
    filename = 'C:\\Users\\123456\\Downloads\\project-3-at-2022-09-17-22-08-efe24f5d.json'
    # read the file in variable
    data = read_json_file(filename)
    # print the data in pretty format
    print(json.dumps(data[1], indent=4))
    # print(data)
    #
    # #get size of list
    # size = len(data)
    # print(size)
    # # get the keys of the dictionary
    #iterate over list and get the key labels
    for i in range(len(data)):
        label = data[i]['label']
        image_name = data[i]['image']

        # get only file name from path and remove extension
        image_name = image_name.split('/')[-1].split('.')[0]
        # print(image_name)

        # print pretty format
        # print(json.dumps(data[i]['label'], indent=4))
        # from key label   get the value x, y, width, height, rotation
        #iterate over label and
        for element in label:
            # print x, y, width, height, rotation from element dictionary
            # print(element)
            # print(element['x'], element['y'], element['width'], element['height'], element['rotation'])

            orig_height = int(element['original_height'])
            orig_width = int(element['original_width'])
            x = float(element['x'])
            y = float(element['y'])
            w = float(element['width'])
            h = float(element['height'])
            theta = float(element['rotation'])
            # print(x, y,w,h,theta,orig_width,orig_height)
            # print(x/orig_width, y/orig_height, w/orig_width, h/orig_height, theta*pi/180)
            # crete generator if element["rectanglelabels"] is "ripe" then 1, if unripe then 2, else 0
            if element["rectanglelabels"][0] == 'ripe':
                out= '{} {} {} {} {} {}'.format(1,x/orig_width, y/orig_height, w/orig_width, h/orig_height, radians(theta))
            elif element["rectanglelabels"][0] == 'unripe':
                out= '{} {} {} {} {} {}'.format(2,x/orig_width, y/orig_height, w/orig_width, h/orig_height, radians(theta))
            elif element["rectanglelabels"][0] == 'bad':
                out= '{} {} {} {} {} {}'.format(0,x/orig_width, y/orig_height, w/orig_width, h/orig_height, radians(theta))
            # print the output
            # print(out)
            # save this output in text file named as the image name
            #open text file and write the output
# get path of current file and create a new file with same name as image name
            dir_name= 'D:/textfiles for R yolo'#os.path.dirname(os.path.realpath(__file__))
            txt_file = os.path.join(dir_name, image_name + '.txt')
            # print('txt file name:', txt_file)
            with open(txt_file, 'a') as f:
                # if element["rectanglelabels"][0] == 'ripe':
                #     f.write('{} {} {} {} {} {}'.format(1, element['x'] / 100, element['y'] / 100, element['width'] / 100, element['height'] / 100,
                #           element['rotation']))
                # elif element["rectanglelabels"][0] == 'unripe':
                #     f.write('{} {} {} {} {}'.format(2, element['x'] / 100, element['y'] / 100, element['width'] / 100, element['height'] / 100,
                #           element['rotation']))
                # else:
                #     f.write('{} {} {} {} {}'.format(0, element['x'] / 100, element['y'] / 100, element['width'] / 100, element['height'] / 100,
                #           element['rotation']))
                # write the output in text file
                f.write(out)
                f.write('\n')
                f.close()





    keys = data[1].keys()
    # print(keys)



'''
def read_json_file(filename):
    import json
    with open(filename, 'r') as f:
        data = json.load(f)
    return json.dumps(data, indent=4, sort_keys=True)
'''