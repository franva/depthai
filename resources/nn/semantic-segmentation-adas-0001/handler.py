import cv2
import numpy as np
from pprint import pprint

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
    def __init__(self) -> None:
        self.COLORS.append(self.VOID)
        self.COLORS.append(self.ROAD)
        self.COLORS.append(self.SIDEWALK)
        self.COLORS.append(self.BUILDING)
        self.COLORS.append(self.WALL)
        self.COLORS.append(self.FENCE)
        self.COLORS.append(self.POLE)
        self.COLORS.append(self.TRAFFIC_LIGHT)
        self.COLORS.append(self.TRAFFIC_SIGN)
        self.COLORS.append(self.VEGETATION)
        self.COLORS.append(self.TRAIN)
        self.COLORS.append(self.SKY)
        self.COLORS.append(self.PERSON)
        self.COLORS.append(self.RIDER)
        self.COLORS.append(self.CAR)
        self.COLORS.append(self.TRUCK)
        self.COLORS.append(self.BUS)
        self.COLORS.append(self.TRAIN)
        self.COLORS.append(self.MOTORCYCLE)
        self.COLORS.append(self.BICYCLE)

        self.COLOR_MAP['VOID'] = self.VOID
        self.COLOR_MAP['ROAD'] = self.ROAD
        self.COLOR_MAP['SIDEWALK'] = self.SIDEWALK
        self.COLOR_MAP['BUILDING'] = self.BUILDING
        self.COLOR_MAP['WALL)'] = self.WALL
        self.COLOR_MAP['FENCE)'] = self.FENCE
        self.COLOR_MAP['POLE)'] = self.POLE
        self.COLOR_MAP['TRAFFIC_LIGHT)'] = self.TRAFFIC_LIGHT
        self.COLOR_MAP['TRAFFIC_SIGN)'] = self.TRAFFIC_SIGN
        self.COLOR_MAP['VEGETATION)'] = self.VEGETATION
        self.COLOR_MAP['TRAIN)'] = self.TRAIN
        self.COLOR_MAP['SKY)'] = self.SKY
        self.COLOR_MAP['PERSON)'] = self.PERSON
        self.COLOR_MAP['RIDER)'] = self.RIDER
        self.COLOR_MAP['CAR)'] = self.CAR
        self.COLOR_MAP['TRUCK)'] = self.TRUCK
        self.COLOR_MAP['BUS)'] = self.BUS
        self.COLOR_MAP['TRAIN)'] = self.TRAIN
        self.COLOR_MAP['MOTORCYCLE)'] = self.MOTORCYCLE
        self.COLOR_MAP['BICYCLE)'] = self.BICYCLE


def decode(nn_manager, packet):
    pprint(vars(packet))
    
    data = np.squeeze(to_tensor_result(packet)["Output/Transpose"])
    class_colors = ColorMap.COLORS
    class_colors = np.asarray(class_colors, dtype=np.uint8)

    output_colors = np.take(class_colors, data, axis=0)
    return output_colors


def draw(nn_manager, data, frames):
    if len(data) == 0:
        return

    for name, frame in frames:
        if name == nn_manager.source:
            cv2.addWeighted(frame, 1, data, 0.2, 0, frame)
