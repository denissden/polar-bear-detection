import json
import cv2
import os
import numpy as np

import utils


PATH = "TEST IMAGES/withBears"
with open("locations.json") as f:
    data = json.load(f)


def main():
    loader = utils.loader(PATH, 0, None, data)
    c = 0

    with open("kernel.txt", "r") as f:
        kernel = np.fromfile(f).reshape(64, 64, 3)
    kernel = kernel / (np.sum(kernel) if np.sum(kernel) != 0 else 1)

    kernels = []
    for i in range(32, 100, 16):
        k = cv2.resize(kernel, (i, i))
        k / (np.sum(k) if np.sum(k) != 0 else 1)
        kernels.append(k)

    cv2.imshow("3", kernel)

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

        marked = []
        tiler = utils.tiler(img, 512, 512)
        for tile in tiler:

            if tile.has_feature or False:
                b, g, r = cv2.split(tile.img)
                b = utils.lossy_normalize(b, 0.01, 0.0001)
                g = utils.lossy_normalize(g, 0.00001, 0.0001)
                r = utils.lossy_normalize(r, 0.00001, 0.0001)
                tile.img = cv2.merge((b, g, r))

                diff = cv2.add(cv2.subtract(r, b), cv2.subtract(g, b))
                diff = utils.lossy_normalize(diff, 0.8, 0.0001, normalize=False)

                values = utils.posterize_counter(diff, 32)
                max_value = utils.cut_colors(values, 0.000001, reverse=True)
                print("mark")
                min_value = utils.cut_colors(values, 0.8)

                marked.append((tile, max_value - np.average(diff), utils.cut_colors(values, 0.01, reverse=True)))
                print(max_value, min_value)
                cv2.imshow("img", tile.img)
                # cv2.imshow("rgb", cv2.hconcat((r, g, b)))
                cv2.imshow("diff", diff)
                cv2.waitKey(0)
                # cv2.destroyAllWindows()
        continue
        print(len(marked))
        cv2.waitKey(0)
        marked = sorted(marked, key=lambda x: x[1], reverse=True)
        print([m[1] for m in marked])
        for m, s, v in marked:
            mi, ma = utils.minmax(m.img)
            # _, th = cv2.threshold(m.img, (mi + ma) / 2, 255, cv2.THRESH_BINARY)
            kernel = np.ones(9).reshape(3, 3)
            # th = cv2.erode(m.img, kernel)
            # th = cv2.dilate(th, kernel)
            cv2.imshow("marked", m.img)
            cv2.waitKey(0)


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
