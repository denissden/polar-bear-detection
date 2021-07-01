import json
import cv2
import os
import numpy as np
from scipy.signal import find_peaks

import utils


PATH = "TEST IMAGES/withBears"
with open("locations.json") as f:
    data = json.load(f)


def main():
    loader = utils.loader(PATH, 5, None, data)
    c = 0

    for img in loader:
        def filter_(img_):
            print(utils.posterize_counter(img_, 16))
            b, g, r = cv2.split(img_)

            def process(i):
                mi, ma = utils.minmax(i)
                print(mi, ma, (mi + ma) // 2)
                i = np.clip(i, mi, (mi + ma) // 2,)
                return i

            r = process(r)
            g = process(g)
            b = process(b)

            img_ = cv2.merge((b, g, r))
            return img_
        def prepare(img_, min_score):
            colors = utils.posterize_counter(img_, 8)
            min_color = utils.cut_colors(colors, 0.1)
            max_color = utils.cut_colors(colors, 0.1, True)

            score = max_color - min_color
            print(score)
            if score < min_score:
                return

            clipped = np.clip(img_, int(min_color), int(max_color))
            norm = cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX)
            return norm
        def temperature(r_, g_, b_):
            rg = r_ + g_
            _t = rg // 2 - b_ + 10
            return _t
        temp = np.vectorize(temperature)

        # max_filtered = None
        # for ker in kernels[::-1]:
        #     filtered = cv2.filter2D(img.img, -1, ker)
        #     if max_filtered is None:
        #         max_filtered = filtered
        #     max_filtered = cv2.max(filtered, max_filtered)
        #
        #     cv2.imshow("f", filtered)
        #     cv2.waitKey()
        #
        # filtered = max_filtered
        #
        # x, y = img.data[0]["pos"]
        # w, h = img.data[0]["size"]
        # cv2.rectangle(filtered, (x, y), (x + w, y + h), (0, 233, 0), 4)
        # cv2.imshow("img", img.img)
        #
        # for i in range(6):
        #     y, x = filtered.shape[:2]
        #     filtered = cv2.pyrDown(filtered, dstsize=(x // 2, y // 2))
        #     b, g, r = cv2.split(filtered)
        #     edges = cv2.Canny(b, 40, 70)
        #     t = np.clip(temp(r, g, b), 0, 255).astype(np.uint8)
        #     cv2.imshow(f"level {i}", t)
        #
        # cv2.waitKey()

        def white_balance(img_):
            result = cv2.cvtColor(img_, cv2.COLOR_BGR2LAB)
            avg_a = np.average(result[:, :, 1])
            avg_b = np.average(result[:, :, 2])
            result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
            result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
            result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
            return result

        img.img = white_balance(img.img)
        # b, g, r = cv2.split(img.img)
        # b = utils.lossy_normalize(b, 0.01, 0.1)
        # g = utils.lossy_normalize(g, 0.01, 0.0001)
        # r = utils.lossy_normalize(r, 0.01, 0.0001)
        # img.img = cv2.merge((b, g, r))

        # cv2.imshow("big", img.img)
        # diff = cv2.subtract(r, b) + cv2.subtract(g, b)
        # img.img = diff
        # cv2.imshow("diff_", diff)

        img.img = cv2.medianBlur(img.img, 11)
        processed = img.img[:, :, 1].copy()


        all_tiles = []
        tilex, tiley = 512, 512
        tiler = utils.tiler(img, tilex, tiley)
        for tile in tiler:

            if tile.has_feature or True:
                b, g, r = cv2.split(tile.img)
                total_range = 0
                b, (mi, ma) = utils.lossy_normalize(b, 0.00001, 0.0001, min_range=20, normalize=True)
                total_range += ma - mi
                g, (mi, ma) = utils.lossy_normalize(g, 0.00001, 0.0001, min_range=20, normalize=True)
                total_range += ma - mi
                r, (mi, ma) = utils.lossy_normalize(r, 0.00001, 0.0001, min_range=20, normalize=True)
                total_range += ma - mi
                print("-------", total_range)
                tile.img = cv2.merge((b, g, r))

                diff = cv2.add(cv2.subtract(r, b), cv2.subtract(g, b))
                # diff, _ = utils.lossy_normalize(diff, 0.8, 0.0001, normalize=False, subtract=False)

                tile.processed = diff
                tile.processed_data["range"] = total_range
                colors = utils.posterize_counter(diff, levels=64)
                tile.processed_data["colors"] = colors
                tile.processed_data["mode"] = utils.cut_colors(colors, 0.5)
                tile.processed_data["max"] = utils.cut_colors(colors, 0.000001, reverse=True)
                all_tiles.append(tile)

                x, y = tile.pos

                if tile.has_feature:
                    print("feature")
                    print(tile.processed_data)

                # img.img[y:y+tiley, x:x+tilex] = tile.img
                mode = tile.processed_data["mode"]
                # cv2.putText(diff,
                #             str(int(mode)),
                #             (0, 128),
                #             cv2.FONT_HERSHEY_SIMPLEX,
                #             1.5,
                #             (255 - mode,),
                #             cv2.LINE_4)
                # cv2.putText(diff,
                #             str(int(tile.processed_data["max"])),
                #             (0, 90),
                #             cv2.FONT_HERSHEY_SIMPLEX,
                #             1.5,
                #             (255 - mode,),
                #             cv2.LINE_4)
                processed[y:y + tiley, x:x + tilex] = diff

                # cv2.imshow("tile", diff)
                # cv2.waitKey()

        all_tiles = filter(lambda x: x.processed_data["mode"] < 10, all_tiles)
        all_tiles = filter(lambda x: x.processed_data["max"] > 60, all_tiles)
        # all_tiles = filter(lambda x: x.processed_data["mode"] > 10, all_tiles)
        # all_tiles = filter(lambda x: x.processed_data["max"] < 60, all_tiles)
        for i in all_tiles:
            x_, y_ = i.pos
            processed[y_:y_+tiley, x_:x_+tilex] = find_bear_shape(i)
            cv2.putText(processed,
                        "USED",
                        (x_, y_ + 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (255,),
                        cv2.LINE_4)


        for d in img.data:
            x, y = d["pos"]
            w, h = d["size"]
            cv2.rectangle(img.img, (x, y), (x + w, y + h), (0, 0, 255), 4)

        sy, sx = img.img.shape[:2]
        cv2.imshow("img", cv2.resize(img.img, (sx // 4, sy // 4)))
        cv2.imshow("diff", cv2.resize(processed, (sx // 4, sy // 4)))
        cv2.waitKey(0)


kernel = np.ones((32, 32), np.float_) * -1
cv2.circle(kernel, (16, 16), 12, 1, thickness=-1)
kernel = kernel / (np.sum(kernel) if np.sum(kernel) != 0 else 1) * -1
print(*kernel)


def find_bear_shape(tile: utils.DataTile):
    pyr = utils.ImagePyramid(tile.processed)
    level_0 = cv2.filter2D(tile.processed, -1, kernel)
    all_levels_filtered = [level_0]
    for i in range(3):
        next_level = next(pyr)
        next_filtered = cv2.filter2D(next_level, -1, kernel)
        all_levels_filtered.append(next_filtered)
    res_tile = np.zeros(tile.processed.shape, np.float_)
    if tile.has_feature or True:
        c = 0
        for i in all_levels_filtered:
            c += 1
            res_tile += cv2.resize(i, tile.processed.shape[:2][::-1])
            cv2.imshow(f"tile{c}", i)
        cv2.imshow(f"res", res_tile)
        # cv2.waitKey()
    return res_tile




if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
