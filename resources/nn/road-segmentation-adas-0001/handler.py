import cv2
import numpy as np

from depthai_helpers.utils import to_tensor_result


def decode(nn_manager, packet):
    [print(f"Layer name: {l.name}, Type: {l.dataType}, Dimensions: {l.dims}") for l in packet.getAllLayers()]
    data = np.squeeze(to_tensor_result(packet)["Output/Transpose"])
    class_colors = [[0,0,0],  [0,255,0]]
    class_colors = np.asarray(class_colors, dtype=np.uint8)

    output_colors = np.take(class_colors, data, axis=0)
    # print('decode() is called')
    return output_colors


def draw(nn_manager, data, frames):
    if len(data) == 0:
        return
    # print('draw() is called')
    for name, frame in frames:
        if name == nn_manager.source:
            cv2.addWeighted(frame, 1, data, 0.2, 0, frame)
