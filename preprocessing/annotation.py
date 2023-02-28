import json
import os

import numpy as np
# import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


path_to_save = "/home/francesco/PycharmProjects/deep_movement_primitives-main/images/"


if __name__ == "__main__":

    json_file = "/home/francesco/Downloads/via_project_23Feb2023_17h40m_json.json"
    f = open(json_file)
    segmentation = json.load(f)
    key = list(segmentation.keys())[0]
    x = int((segmentation[key]["regions"][0]["shape_attributes"]["x"]))
    y = int((segmentation[key]["regions"][0]["shape_attributes"]["y"]))
    width = int((segmentation[key]["regions"][0]["shape_attributes"]["width"]))
    height = int((segmentation[key]["regions"][0]["shape_attributes"]["height"]))
    filename = (segmentation[key]["filename"])

    strawberry_file = "/home/francesco/Downloads/proper_test.png"

    image = Image.open(strawberry_file)
    pixel_map = image.load()
    for i in range(x, x+width):     # columns
        for j in range(y, y+height):    # rows
            pixel_map[i, j] = (int(255), int(255), int(255))

    image.save(path_to_save + "proper_robot_test_4.png")
