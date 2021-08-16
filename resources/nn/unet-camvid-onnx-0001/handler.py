from logging import exception
import cv2
import numpy as np
# from pprint import pprint
from numpy import save

from depthai_helpers.utils import to_tensor_result

class ColorMap:        
    SKY=[28,51,71]
    BUILDING=[28,28,28]
    POLE=[60,60,60]
    ROAD=[50,25,50]
    PAVEMENT=[95,14,91]
    FENCE=[74,60,60]
    VEHICLE=[0,0,56]
    PEDESTRIAN=[86,8,23]
    BIKE=[47,5,13]
    UNLABELED = [17,18,21]
    TREE=[40,40,61]
    SIGNSYMBOL=[86,86,0]

    COLORS = []
    COLORS_BGR = []
    COLOR_MAP = {}

    # the sequence of colors in this arrar matters!!! as it maps to the prediction classes    
    COLORS.append(SKY)
    COLORS.append(BUILDING)
    COLORS.append(POLE)
    COLORS.append(ROAD)
    COLORS.append(PAVEMENT)
    COLORS.append(TREE)
    COLORS.append(SIGNSYMBOL)
    COLORS.append(FENCE)
    COLORS.append(VEHICLE)
    COLORS.append(PEDESTRIAN)
    COLORS.append(BIKE)
    COLORS.append(UNLABELED)

    for color in COLORS:
        np_color = np.array(color)
        COLORS_BGR.append(np_color[[2,1,0]])

    COLOR_MAP['SKY)'] = SKY
    COLOR_MAP['UNLABELED'] = UNLABELED
    COLOR_MAP['ROAD'] = ROAD
    COLOR_MAP['PAVEMENT'] = PAVEMENT
    COLOR_MAP['BUILDING'] = BUILDING
    COLOR_MAP['TREE)'] = TREE
    COLOR_MAP['FENCE)'] = FENCE
    COLOR_MAP['POLE)'] = POLE
    COLOR_MAP['SIGNSYMBOL)'] = SIGNSYMBOL
    COLOR_MAP['VEHICLE)'] = VEHICLE
    COLOR_MAP['PEDESTRIAN)'] = PEDESTRIAN
    COLOR_MAP['BIKE)'] = BIKE

def decode(nn_manager, packet):

    # [print(f"Layer name: {l.name}, Type: {l.dataType}, Dimensions: {l.dims}") for l in packet.getAllLayers()]
    # Layer name: 206, Type: DataType.FP16, Dimensions: [1, 12, 368, 480]

    layer_name = '206'

    data = np.squeeze(to_tensor_result(packet)[layer_name]).transpose()
    # data.shape, data[0,0,0]
    # (12, 368, 480), 0.07977294921875

    if data is None:
        raise exception('decode does not get data from packet')

    # np.argmax(data[:, 0 ,0])
    # use argmax to get the index of the max value in the 12 dimensions 
    # map to the respective color according to the index
    data = np.argmax(data, axis=0)
    if data.shape != (368, 480):
        raise exception('unexpected shape of data from decode() method in handler.py');

    class_colors = ColorMap.COLORS
    class_colors = np.asarray(class_colors, dtype=np.uint8)

    output_colors = np.take(class_colors, data, axis=0)
    return output_colors

def draw(nn_manager, data, frames):
    if len(data) == 0:
        return

    for name, frame in frames:
        if name == nn_manager.source:
            # cv2.addWeighted(frame, 0.9, data, 0.9, 0, frame)

            # what's the shape of data and frames
            # img = cv2.addWeighted(frame, 1, data, 0.5, 0)
            cv2.addWeighted(frame, 1, data, 0.5, 0, frame)

            # cv2.imwrite('seg_color.jpg', data)
            # print('segmentation is drawn on the video.')

            # TODO: try to swap color channel to BGR
            # TODO: try to move the color channel to the 1st dimension -> 3, 368,480, or 3,480,368
            # return img

