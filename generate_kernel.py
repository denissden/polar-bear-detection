import json
import cv2
import os
import numpy as np

import utils


PATH = "cropped_bears"
with open("locations.json") as f:
    data = json.load(f)


def main():
    shape = (64, 64, 3)
    kernel = np.zeros(shape, np.float)

    loader = utils.loader(PATH)

    sizes = []
    c = 0
    for img in loader:
        im = img.img

        sizes.append(im.shape)

        resized = cv2.resize(im, shape[:2])
        kernel += cv2.normalize(resized.astype(np.float), 0., 1.)
        # kernel += resized.astype(np.float) / 255

        c += 1

    print(*sorted(sizes), sep="\n")

    cv2.imshow("k", kernel / c)
    cv2.waitKey()

    # with open("kernel.txt", "+w") as f:
    #     kernel.tofile(f)


if __name__ == '__main__':
    main()