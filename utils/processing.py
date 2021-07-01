import cv2
import numpy as np
import numba


@numba.jit
def minmax(x: np.ndarray):
    conv = x.ravel()
    maximum = conv[0]
    minimum = conv[0]
    for i in conv[1:]:
        if i > maximum:
            maximum = i
        elif i < minimum:
            minimum = i
    return minimum, maximum


@numba.jit
def posterize_counter(x: np.ndarray, levels: int = 8, n_colors: int = 256):
    conv = x.ravel()
    colors = [0. for _ in range(levels)]
    div = n_colors // levels
    for i in conv:
        colors[i // div] += 1
    return np.array([i / len(conv) for i in colors])


@numba.jit
def cut_colors(colors: np.array, target: float, reverse=False, n_colors: float = 255.):
    accuracy = len(colors)
    if reverse:
        target = 1 - target

    if target == 0 or target == 1:
        return n_colors * target

    sum_ = 0.
    for i in range(accuracy):
        c = colors[i]
        if sum_ + c < target:
            sum_ += c
        else:
            next_val = sum_ + c
            diff_val = next_val - sum_
            left = target - sum_
            mult = left / diff_val
            res = i * (1 - mult) + (i + 1) * mult
            factor = n_colors / accuracy
            return res * factor


def lossy_normalize(img: np.ndarray,
                    min_cut: float,
                    max_cut: float,
                    accuracy: int = 8,
                    normalize=True,
                    subtract=True,
                    min_range: float = 0):
    colors = posterize_counter(img, accuracy)
    min_color = cut_colors(colors, min_cut)
    max_color = cut_colors(colors, max_cut, True)

    ret_data = (min_color, max_color)

    if min_range and max_color - min_color < min_range:
        max_color = max_color + min_range / 2
        min_color_decrement = min_range / 2
        if int(max_color) > 255:
            max_color = 255
            min_color_decrement += max_color - 255.

        min_color -= min_color_decrement
        if min_color < 0:
            min_color = 0

    clipped = np.clip(img, int(min_color), int(max_color))
    if subtract:
        clipped -= np.uint8(min_color)
    if normalize:
        norm = cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX)
        return norm, ret_data
    return clipped, ret_data

