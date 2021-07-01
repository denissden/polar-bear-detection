import dataclasses
import json
import cv2
import os
import numpy as np
from scipy.signal import find_peaks

import utils


PATH = "TEST IMAGES/withBears"  # путь к папке
with open("locations.json") as f:
    data = json.load(f)


def main():
    loader = utils.loader(PATH, 0, None, data)
    # loader = utils.loader(PATH, 0, None) # разкомментировать
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
            # norm = cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX)
            return clipped


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
        imgcopy = img.img.copy()

        all_tiles = []
        tilex, tiley = 512, 512
        tiler = utils.tiler(img, tilex, tiley)

        # FIRST STEP - split image into tiles and process each one individually
        for tile in tiler:
            b, g, r = cv2.split(tile.img)
            total_range = 0
            b, (mi, ma) = utils.lossy_normalize(b, 0.00001, 0.0001, min_range=20, normalize=True)
            total_range += ma - mi
            g, (mi, ma) = utils.lossy_normalize(g, 0.00001, 0.0001, min_range=20, normalize=True)
            total_range += ma - mi
            r, (mi, ma) = utils.lossy_normalize(r, 0.00001, 0.0001, min_range=20, normalize=True)
            total_range += ma - mi
            tile.img = cv2.merge((b, g, r))

            diff = cv2.add(cv2.subtract(r, b), cv2.subtract(g, b))
            # diff, _ = utils.lossy_normalize(diff, 0.8, 0.0001, normalize=False, subtract=False)

            tile.processed = diff
            tile.processed_data["range"] = total_range
            colors = utils.posterize_counter(diff, levels=64)
            tile.processed_data["colors"] = colors
            tile.processed_data["median"] = utils.cut_colors(colors, 0.5)
            tile.processed_data["max"] = utils.cut_colors(colors, 0.000001, reverse=True)
            all_tiles.append(tile)

            x, y = tile.pos

            # if tile.has_feature:
            #     print("feature")
            #     print(tile.processed_data)

            # img.img[y:y+tiley, x:x+tilex] = tile.img
            median = tile.processed_data["median"]
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
            # img.img[y:y + tiley, x:x + tilex] = tile.img

            # cv2.imshow("tile", diff)
            # cv2.waitKey()

        # filter out tiles that are not likely to have a bear inside them
        filtered_tiles_1 = filter(lambda x: x.processed_data["median"] < 10, all_tiles)
        filtered_tiles_1 = filter(lambda x: x.processed_data["max"] > 60, filtered_tiles_1)

        filtered_tiles_2 = []
        for tile in filtered_tiles_1:
            x_, y_ = tile.pos
            filtered_tile = find_bear_shape(tile.processed)
            filtered_tile = np.clip(filtered_tile, 0, 255).astype(np.uint8)

            filtered_tile, (mi, ma) = utils.lossy_normalize(filtered_tile, 0.9, 0, normalize=False)
            filtered_tile = cv2.blur(filtered_tile, (31, 31))
            blur_min, blur_max = utils.minmax(filtered_tile)

            small_data = cv2.resize(filtered_tile, (8, 8), interpolation=cv2.INTER_AREA)
            if tile.has_feature and False:
                cv2.imshow("small", small_data)
                cv2.waitKey()
            max_val = np.max(small_data)

            tile.processed = cv2.normalize(tile.processed, None, 0, 255)

            tile.processed_data["2max"] = max_val
            tile.processed_data["2norm_min"] = blur_min
            tile.processed_data["2norm_max"] = blur_max
            # tile.processed_data["2dark"] = dark_amount

            processed[y_:y_+tiley, x_:x_+tilex] = filtered_tile
            if False:  # show text
                cv2.putText(processed,
                            "USED",
                            (x_, y_ + 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5,
                            (255,),
                            cv2.LINE_4)
                cv2.putText(processed,
                            str(int(max_val)),
                            (x_, y_ + 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5,
                            (255,),
                            cv2.LINE_4)


            tile.processed = filtered_tile
            filtered_tiles_2.append(tile)

        filtered_tiles_2_extracted = filter(lambda x: x.processed_data["2max"] > 5, filtered_tiles_2)

        filtered_tiles_3 = []
        for tile in filtered_tiles_2_extracted:
            x_, y_ = tile.pos
            contours, _ = cv2.findContours(
                tile.processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)

            tile.processed_data["contours"] = []
            tile.processed_data["areas"] = []
            areas = []
            for contour in contours:
                # cv2.approxPloyDP() function to approximate the shape
                area = cv2.contourArea(contour)
                areas.append(area)

                cv2.drawContours(tile.img, [contour], 0, (0, 255, 0), 5)

            tile.processed_data["max_area"] = max(areas)
            tile.processed_data["areas"] = areas
            tile.processed_data["contours"] = contours
            img.img[y_:y_ + tiley, x_:x_ + tilex] = tile.img

            filtered_tiles_3.append(tile)

        points = []   # point: (x, y), area
        for tile in filtered_tiles_3:
            x_, y_ = tile.pos

            tile_range = tile.processed_data["range"]
            print("range", tile_range)
            tile_max = tile.processed_data["2norm_max"]

            for c, a in zip(tile.processed_data["contours"], tile.processed_data["areas"]):
                if a > 0:
                    M = cv2.moments(c)
                    center_x, center_y = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                    processed_color = tile.processed[center_y, center_x]

                    global_x, global_y = x_ + center_x, y_ + center_y
                    color = imgcopy[global_y, global_x]

                    dist = utils.distance_from_center(center_x, center_y, c)

                    points.append(utils.Point(pos=(global_x, global_y),
                                              tile_pos=(x_, y_),
                                              area=a,
                                              color=color,
                                              processed_color=processed_color,
                                              range_=tile_range,
                                              max_=tile_max,
                                              distance=dist,
                                              contour=c))

                    blur_min = tile.processed_data["2norm_min"]
                    blur_max = tile.processed_data["2norm_max"]

        points = filter(lambda x: x.distance > 20, points)
        points = sorted(points, key=lambda x: x.processed_color, reverse=True)

        factor = int(255 / len(points) * 2)
        c = 255
        # for p in points[:1]:
        for p in points[:(len(points) + 1) // 2]:
            # cv2.circle(imgcopy, p.pos, 8, (c, int(p.processed_color), 0), thickness=-1)
            # c -= factor

            size = 50
            x_, y_ = p.pos
            minx = (x_ - size) if (x_ - size) > 0 else 0
            miny = (y_ - size) if (y_ - size) > 0 else 0
            cropped = imgcopy[miny:y_ + size, minx:x_ + size]
            light_min, light_max = utils.minmax(cropped)
            if light_max - light_min < 20:
                continue
            else:
                cont = p.contour
                tx, ty = p.tile_pos
                cont[:, 0, 0] += tx
                cont[:, 0, 1] += ty
                print(cont)
                cv2.drawContours(imgcopy, [cont], 0, (0, 255, 0), 5)
                b, g, r = cv2.split(cropped.astype(np.int16))
                total_range = 0
                b, (mi, ma) = utils.lossy_normalize(b, 0.00001, 0.0001, min_range=20, normalize=True)
                g, (mi, ma) = utils.lossy_normalize(g, 0.00001, 0.0001, min_range=20, normalize=True)
                r, (mi, ma) = utils.lossy_normalize(r, 0.00001, 0.0001, min_range=20, normalize=True)
                diff = cv2.add(cv2.subtract(r, b), cv2.subtract(g, b))
                break


            def temperature(r_, g_, b_):
                rg = r_ + g_
                _t = rg // 2 - b_ + 10
                return _t * (rg / 255)

            temp = np.vectorize(temperature)

            temp_map = np.clip(temp(r, g, b), 0, 255).astype(np.uint8)

            print(total_range)
            cv2.imshow("crp", cropped)
            # cv2.imshow("temp", temp_map)
            cv2.waitKey()

        for d in img.data:
            x, y = d["pos"]
            w, h = d["size"]
            cv2.rectangle(imgcopy, (x, y), (x + w, y + h), (0, 0, 255), 4)

        sy, sx = img.img.shape[:2]
        cv2.imshow("img", cv2.resize(imgcopy, (sx // 4, sy // 4)))
        # cv2.imshow("diff", cv2.resize(processed, (sx // 4, sy // 4)))
        # cv2.imshow("copy", imgcopy)
        cv2.waitKey(0)


kernel = np.ones((32, 32), np.float_) * -1
cv2.circle(kernel, (16, 16), 12, 1, thickness=-1)
kernel = kernel / (np.sum(kernel) if np.sum(kernel) != 0 else 1) * -1
print(*kernel)


def find_bear_shape(tile: np.ndarray):
    pyr = utils.ImagePyramid(tile.astype(np.float))
    level_0 = cv2.filter2D(tile, -1, kernel)
    all_levels_filtered = [level_0]
    for i in range(3):
        next_level = next(pyr)
        next_filtered = cv2.filter2D(next_level, -1, kernel)
        all_levels_filtered.append(next_filtered)
    res_tile = np.zeros(tile.shape, np.float_)

    c = 0
    for i in all_levels_filtered:
        c += 1
        res_tile += cv2.resize(i, tile.shape[:2][::-1])
    return res_tile

if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
