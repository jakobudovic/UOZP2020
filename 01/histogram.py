import numpy as np
import csv
import matplotlib.pyplot as plt

file_name = "eurovision-finals-1975-2019.csv"
f = open(file_name, "rt")
values = csv.reader(f)
header_line = next(values)  # skip header cell
values = np.array(list(values))  # values used for unique countries extraction
print("header_line:", header_line)

points = values[:,4]

points1 = [l for l in points if l != 0]
# points1 = [x if x % 2 else x * 100 for x in range(1, 10) ]

points2 = []

for l in points:
    # print(type(int(l)))
    if int(l) > 0:
      points2.append(l)

print("points2:", points2)
# print(values[:,4])
# print(len(values[:,4]))

# plt.bar(points, np.arange(len(points)))
plt.hist(points, bins=11)
# plt.bar(points, np.arange(len(points)))
plt.show()

print(len(points))
