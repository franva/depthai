from logging import exception
import cv2
import numpy as np
# from pprint import pprint
from numpy import save

from depthai_helpers.utils import to_tensor_result

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

