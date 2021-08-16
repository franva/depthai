from logging import exception
import cv2
import numpy as np
# from pprint import pprint
from numpy import save

from depthai_helpers.utils import to_tensor_result

class ColorMap:        
    VOID = [17,18,21]
    ROAD=[50,25,50]
    SIDEWALK=[95,14,91]
    BUILDING=[28,28,28]
    WALL=[40,40,61]
    FENCE=[74,60,60]
    POLE=[60,60,60]
    TRAFFIC_LIGHT=[98,67,12]
    TRAFFIC_SIGN=[86,86,0]
    VEGETATION=[42,56,13]
    TERRAIN=[60,98,60]
    SKY=[28,51,71]
    PERSON=[86,8,23]
    RIDER=[100,0,0]
    CAR=[0,0,56]
    TRUCK=[0,0,28]
    BUS=[0,24,39]
    TRAIN=[0,31,39]
    MOTORCYCLE=[0,0,90]
    BICYCLE=[47,5,13]

    COLORS = []
    COLOR_MAP = {}
    
    COLORS.append(VOID)
    COLORS.append(ROAD)
    COLORS.append(SIDEWALK)
    COLORS.append(BUILDING)
    COLORS.append(WALL)
    COLORS.append(FENCE)
    COLORS.append(POLE)
    COLORS.append(TRAFFIC_LIGHT)
    COLORS.append(TRAFFIC_SIGN)
    COLORS.append(VEGETATION)
    COLORS.append(TRAIN)
    COLORS.append(SKY)
    COLORS.append(PERSON)
    COLORS.append(RIDER)
    COLORS.append(CAR)
    COLORS.append(TRUCK)
    COLORS.append(BUS)
    COLORS.append(TRAIN)
    COLORS.append(MOTORCYCLE)
    COLORS.append(BICYCLE)

    COLOR_MAP['VOID'] = VOID
    COLOR_MAP['ROAD'] = ROAD
    COLOR_MAP['SIDEWALK'] = SIDEWALK
    COLOR_MAP['BUILDING'] = BUILDING
    COLOR_MAP['WALL)'] = WALL
    COLOR_MAP['FENCE)'] = FENCE
    COLOR_MAP['POLE)'] = POLE
    COLOR_MAP['TRAFFIC_LIGHT)'] = TRAFFIC_LIGHT
    COLOR_MAP['TRAFFIC_SIGN)'] = TRAFFIC_SIGN
    COLOR_MAP['VEGETATION)'] = VEGETATION
    COLOR_MAP['TRAIN)'] = TRAIN
    COLOR_MAP['SKY)'] = SKY
    COLOR_MAP['PERSON)'] = PERSON
    COLOR_MAP['RIDER)'] = RIDER
    COLOR_MAP['CAR)'] = CAR
    COLOR_MAP['TRUCK)'] = TRUCK
    COLOR_MAP['BUS)'] = BUS
    COLOR_MAP['TRAIN)'] = TRAIN
    COLOR_MAP['MOTORCYCLE)'] = MOTORCYCLE
    COLOR_MAP['BICYCLE)'] = BICYCLE


def decode(nn_manager, packet):
    
    # [print(f"Layer name: {l.name}, Type: {l.dataType}, Dimensions: {l.dims}") for l in packet.getAllLayers()]
    layer_name = 'segmentation_output/Squeeze'

    # convert 960x720x1 ==> 960x720x1
    data = np.squeeze(to_tensor_result(packet)[layer_name]).transpose()

    if data is None:
        raise exception('decode does not get data from packet')

    # np.savetxt('packet.csv', np.squeeze(data), delimiter=',')
    # save('data.npy', data)
    class_colors = ColorMap.COLORS
    class_colors = np.asarray(class_colors, dtype=np.uint8)

    # exit(0)
    output_colors = np.take(class_colors, data, axis=0)
    return output_colors


def draw(nn_manager, data, frames):
    if len(data) == 0:
        return

    for name, frame in frames:
        if name == nn_manager.source:
            cv2.addWeighted(frame, 0.9, data, 0.9, 0, frame)
            # return cv2.addWeighted(frame, 1, data, 0.5, 0)
