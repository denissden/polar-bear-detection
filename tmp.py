import os
import cv2
import json

import utils

PATH = "bears"
with open("locations.json") as f:
    data = json.load(f)


def main():
    loader = utils.loader(PATH, data=data)

    # for img in loader:
    #     b, g, r = cv2.split(img.img)
    #     cv2.imshow("rgb", cv2.hconcat((r, g, b)))
    #     cv2.waitKey()

    for k, v in data.items():
        w, h = v[0]["size"]
        print(w * h)


if __name__ == '__main__':
    main()
