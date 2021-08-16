from typing import overload
from depthai_helpers.managers import Previews
from resources.nn.colormap import ColorMap
import cv2
import numpy as np

from depthai_helpers.utils import to_tensor_result


def decode(nn_manager, packet):
    # [print(f"Layer name: {l.name}, Type: {l.dataType}, Dimensions: {l.dims}") for l in packet.getAllLayers()]
    # after squeeze the data.shape is 4,512, 896
    data = np.squeeze(to_tensor_result(packet)["L0317_ReWeight_SoftMax"])
    class_colors = ColorMap.COLORS
    class_colors = np.asarray(class_colors, dtype=np.uint8)

    indices = np.argmax(data, axis=0)
    indices += 8
    output_colors = np.take(class_colors, indices, axis=0)
    return output_colors



def draw(nn_manager, data, frames):
    if len(data) == 0:
        return
    for name, frame in frames:
        if name in (nn_manager.source, 'host'):
            resized = frame
            if frame.shape != data.shape:
                # resize the frame to fit data
                # newsize = data.shape[:2][1],data.shape[:2][0]
                # resized = cv2.resize(frame, newsize)

                # resize the data to fit frame
                newsize = frame.shape[:2][1],frame.shape[:2][0]
                resized = cv2.resize(data, newsize)

            # overlayed = (resized * 0.5 + data * 0.5).astype('uint8')
            # overlayed = cv2.addWeighted(resized, 0.5, data, 0.5, 0)
            overlayed = (frame * 0.5 + resized * 0.5).astype('uint8')

            return overlayed
            # return data

# def draw(nn_manager, data, frames):
#     if len(data) == 0:
#         return
#     for name, frame in frames:
#         if name in (Previews.color.name, Previews.nn_input.name, 'host'):
#             scale_factor = frame.shape[0] / nn_manager.input_size[1]
#             resize_w = int(nn_manager.input_size[0] * scale_factor)
#             resized = cv2.resize(data, (resize_w, frame.shape[0])).astype(data.dtype)
#             offset_w = int(frame.shape[1] - nn_manager.input_size[0] * scale_factor) // 2
#             tail_w = frame.shape[1] - offset_w - resize_w
#             stacked = np.hstack((np.zeros((frame.shape[0], offset_w, 3)).astype(resized.dtype), resized, np.zeros((frame.shape[0], tail_w, 3)).astype(resized.dtype)))
#             cv2.addWeighted(frame, 1, stacked, 0.2, 0, frame)
