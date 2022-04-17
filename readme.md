# Python Project Tomato



## Intro

* This project based on dcnn that find tomatoes from camera intel realsence d455.
* Using pyrealsence, numpy and matplotlib.

### TODO:

check of green boxes side ratio h/w  
enable autoexposure in color image
emitter enable testing
lase gain testing
## Basic classes

* function coords(box,picture,picture,depth_frame,profile)
```python
bad_tomato_coordinates = coords(rectangle_of_tomatoes, qualify_result ,qualify_image,qualify_depth,colorized_depth,profile)
```
## Code
### Alignement of depth and color frames
[code from github](https://github.com/ut-robotics/picr21-team-4meats/blob/0a2f68959e92fb180e8dc32ea1351e628a1b4e30/camera.py#L73):
```python
frames = self.pipeline.wait_for_frames()
aligned_frames = self.align.process(frames)
depth_frame = aligned_frames.get_depth_frame()
color_frame = aligned_frames.get_color_frame()
```

Added explosure and white balance options in 99:

```python
## filter options
    color_sensor = profile.get_device().query_sensors()[1]
    color_sensor.set_option(rs.option.enable_auto_white_balance, True)
    color_sensor.set_option(rs.option.exposure,1000)
    color_sensor.set_option(rs.option.enable_auto_exposure,True)
```


## Emitter enabled

This options you can find in cpp code of intelrealsense src\ds5\advanced_mode\advanced_mode.cpp