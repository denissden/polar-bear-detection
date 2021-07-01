import collections
import json
import cv2
import numpy as np
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Iterable


@dataclass
class DataImage:
    img: np.ndarray
    data: List[dict] = field(default_factory=list)


@dataclass
class DataTile:
    img: np.ndarray
    pos: Tuple[int, int]
    has_feature: bool
    data: List[dict] = field(default_factory=list)
    processed: np.ndarray = field(default=None)
    processed_data: dict = field(default_factory=dict)


@dataclass
class ImagePyramid:
    img: np.ndarray
    levels: List[np.ndarray]
    factor: float = 2

    def __init__(self, image: np.ndarray):
        self.img = image
        self.levels = [image]

    def __next__(self):
        last_level = self.levels[-1]
        sy, sx = last_level.shape[:2]
        next_level = cv2.pyrDown(last_level, dstsize=(int(sx // self.factor), int(sy // self.factor)))
        self.levels.append(next_level)
        return next_level


def loader(path: str, from_: int = 0, to_: int = None, data: dict = None) -> Iterable[DataImage]:
    """
    Loads images from directory and attaches data to them if given

    :param path: path to directory
    :param from_: load images FROM
    :param to_: load images TO
    :param data: dict data containing image names as keys
    :return: generate *DataImage* images
    """
    files = os.listdir(path)
    if from_ >= len(files):
        return
    if not to_ or to_ > len(files):
        to_ = len(files)

    for name in files[from_:to_]:
        img = cv2.imread(os.path.join(path, name))
        d = None
        if data and name in data:
            d = data[name]

        yield DataImage(img, d)


def tiler(data_img: DataImage, tx: int, ty: int, pos_key="pos", size_key="size") -> Iterable[DataTile]:
    """
    Split image into tiles of specified size while keeping data

    :param data_img: image
    :param tx: tile width
    :param ty: tile height
    :param pos_key: position key in image data dict
    :param size_key: size key in image data dict
    :return: generate *DataTile* tiles
    """
    sx, sy = data_img.img.shape[:2]
    for x in range(0, sy, ty):
        for y in range(0, sx, tx):
            # create tile with cropped image and tile position
            tile = DataTile(data_img.img[y:y+ty, x:x+tx], (x, y), has_feature=False)
            for d in data_img.data:
                dx, dy = d[pos_key]
                dw, dh = d[size_key]

                # check feature from data collides with tile area
                if x - dw <= dx <= x + tx and y - dh <= dy <= y + ty:
                    tile.has_feature = True
                    tile.data.append({
                        # account to tile position
                        pos_key: (dx - x, dy - y),
                        # keep size that same
                        size_key: (dw, dh)
                    })
            yield tile
