import numpy as np
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