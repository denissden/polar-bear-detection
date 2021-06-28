import json

import cv2
import os


PATH = "TEST IMAGES/withBears"
with open("locations.json") as f:
    data = json.load(f)


def main():
    for name in os.listdir(PATH):
        img = cv2.imread(PATH + "/" + name)
        print(img.shape)
        d = data[name]
        for i in range(0, len(d), 2):
            x, y = d[i]
            w, h = d[i + 1]
            print(x, y, w, h, name)
            cropped = img[y - 100:y+h + 100, x - 100 :x+w + 100]
            # cropped = img[0:100, 0:100]
            # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 5)
            cv2.imshow("img", cropped)
            cv2.imwrite("bears/" + name, cropped)
            # cv2.waitKey()




if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
