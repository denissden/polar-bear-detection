import os

files = os.listdir("TEST IMAGES/withBears")

res = dict()

for i in files:
    res[i] = [[-1, -1], [-1, -1]]

with open("locations___.json", "w+") as f:
    f.write(str(res).replace("'", '"'))
