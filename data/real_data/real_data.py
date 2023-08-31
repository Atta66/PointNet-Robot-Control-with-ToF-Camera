# import open3d as o3d
# import numpy
# pcd = o3d.io.read_point_cloud("e2.pcd")
# points = numpy.asarray(pcd.points)
# print(len(points))
# o3d.visualization.draw_geometries([pcd])

# import keyboard
# i=0
# while True:

# 	input("press key...")
# 	i += 1
# 	print(i)
from collections import Counter

a = ["20","30"]
b = ["30", "21"]

if Counter(map(str, a)) == Counter(map(str, b)):
	print("yes")





# print(points[1][0])
# pred_poi = 0
# pred_poi_sum = 0
# pred_npoi = 0
# pred_npoi_sum = 0

# for i in range(1024):
#     if classes[np.argmax(label_cloud[i])] == "POI_extension" and pred_poi != 200:
#         pred_poi += 1
#         pred_poi_sum += points[i][0]
#     if classes[np.argmax(label_cloud[i])] == "nPOI_extension" and pred_npoi != 200:
#         pred_npoi += 1
#         pred_npoi_sum += points[i][0]
            
# print("pred_poi = %d, pred_poi_sum = %f, pred_npoi = %d, pred_npoi_sum = %f" % (pred_poi, pred_poi_sum, pred_npoi, pred_npoi_sum))

# pred_poi = 0
# pred_poi_sum = 0
# pred_npoi = 0
# pred_npoi_sum = 0
# for i in range(1024):
#     if classes[np.argmax(label_cloud[i])] == "POI_pliers" and pred_poi != 200:
#         pred_poi += 1
#         pred_poi_sum += points[i][2]
#     if classes[np.argmax(label_cloud[i])] == "nPOI_pliers" and pred_npoi != 200:
#         pred_npoi += 1
#         pred_npoi_sum += points[i][2]
            
# print("pred_poi = %d, pred_poi_sum = %f, pred_npoi = %d, pred_npoi_sum = %f" % (pred_poi, pred_poi_sum, pred_npoi, pred_npoi_sum))
	