# import os.path
# import open3d as o3d
# import sys
# import numpy as np

########################## circle, down and pop #############################################################


# # Read point cloud from PLY
# pcd1 = o3d.io.read_point_cloud("royale.ply")
# points = np.asarray(pcd1.points)
#
# # Sphere center and radius
# center = np.array([0,0,0])
# radius = 0.5
#
# # Calculate distances to center, set new points
# distances = np.linalg.norm(points - center, axis=1)
# pcd1.points = o3d.utility.Vector3dVector(points[distances <= radius])
# #
# #
# # Write point cloud out
# o3d.io.write_point_cloud("out.pcd", pcd1)
# o3d.visualization.draw_geometries([pcd1])
# print(pcd1.points)
# downpcd = pcd1.voxel_down_sample(voxel_size=0.003
#                                 )
# print(downpcd.points)
# o3d.visualization.draw_geometries([downpcd])
# while len(np.asarray(downpcd.points)) > 4096:
#     #print("very much points")
#     downpcd.points.pop()
# o3d.visualization.draw_geometries([downpcd])
# print(downpcd.points)

#############################################################################################################

######################################## remove background ##################################################

# lpcd = o3d.io.read_point_cloud("data/drill_box1-upright/1.ply")
#
#
# i = -1
# pcd = lpcd.voxel_down_sample(voxel_size=0.08
#                                 )
# o3d.visualization.draw_geometries([pcd])
# print(pcd.points)
# a = -1
# while len(pcd.points) > 200:
#     i += 1
#     print("out")
#     if pcd.points[i][2] > 0.2:
#         pcd.points.pop(i)
#         a = len(pcd.points)
#         print(len(pcd.points))
#
#     else:
#         while len(pcd.points) > 200:
#             print("ich here but ", len(pcd.points))
#             pcd.points.pop()
#
# print(pcd.points[1][1])
# print(len(pcd.points))
# o3d.visualization.draw_geometries([pcd])


##############################################################################################################



##################################### assign point cloud #####################################################

# path = "data/drill_box1-upright"
#
# pcd = o3d.io.read_point_cloud(path+"/1.ply")
# pcd = pcd.voxel_down_sample(voxel_size=0.004
#                                  )
#
#
# #new_pcd = np.zeros((4096,3))
#
# num_points = 4096
#
# new_pcd = o3d.geometry.PointCloud()
# new_pcd.points = o3d.utility.Vector3dVector(np.zeros((num_points,3)))
# print(np.asarray(pcd))
# print(np.asarray(new_pcd))
#
#
# a =1
# for i in range(len(pcd.points)):
#     if a == num_points:
#         break
#     elif pcd.points[i][2] < 0.5:
#         new_pcd.points[a][0] = pcd.points[i][0]
#         new_pcd.points[a][1] = pcd.points[i][1]
#         new_pcd.points[a][2] = pcd.points[i][2]
#         a += 1
#         print(a, i)
#
#     # if i == a:
#     #     break
#
#     #else:  # here 8000 fills out but our important points might be at index value above 8000
#
#
#
#
# #print(new_pcd.points[1999][1])
# #o3d.io.write_point_cloud(path+"/wrench10_4096.ply", new_pcd)
# #o3d.visualization.draw_geometries([pcd])
# o3d.visualization.draw_geometries([new_pcd])
#
#
# print(np.asarray(pcd.points).shape)

##############################################################################################################

#####################################traditional distance_filter##############################################

# import os
# from pathlib import Path
# import open3d as o3d
#
# pathtofolder = os.path.join(Path.home(), 'libroyale-4.24.0.1201-LINUX-x86-64Bit/python/files/')
# # print(pathtofolder)
# pathtosave = os.path.join(Path.home(), 'libroyale-4.24.0.1201-LINUX-x86-64Bit/python/filtered/')
#
#
# for i in range(len(os.listdir(pathtofolder))):
#     pcd = o3d.io.read_point_cloud(pathtofolder+str(i)+".pcd")
#     points = np.asarray(pcd.points)
#
#     pcd = pcd.select_by_index(np.where((points[:, 1] < 0.5) & (points[:, 1] > 0.05)
#                                        & (points[:, 0] < 0.2) & (points[:, 0] > -0.2)
#                                        & (points[:, 2] < 0.6) & (points[:, 2] > 0.3)
#                                        )[-1])
#
#     o3d.io.write_point_cloud(pathtosave+str(i)+".pcd", pcd)

##############################################################################################################


######################## circle filter & farthest_point_sampling #############################################

# path = "data/drill_charger_todo/"
# frame = '9'
#
# pcd = o3d.io.read_point_cloud(path+frame+".ply")
# points = np.asarray(pcd.points)
#
# center = np.array([0,0,0])
# radius = 0.5
#
# # Calculate distances to center, set new points
# distances = np.linalg.norm(points - center, axis=1)
# pcd.points = o3d.utility.Vector3dVector(points[distances <= radius])
#
#
#
# #print(new_pcd.points[1999][1])
#
# #o3d.visualization.draw_geometries([pcd])
# #o3d.visualization.draw_geometries([pcd])
#
# #pcd_down = pcd.farthest_point_down_sample(4096)
# #o3d.visualization.draw_geometries([pcd_down])
# o3d.io.write_point_cloud(path+"ed"+frame+".pcd", pcd)
#
# #print(np.asarray(pcd_down.points).shape)
# print(np.asarray(pcd.points).shape)
#


#################################################################################################
########################ply to pcd###############################################################

# file = os.path.abspath('../../../../Users/Atta/Downloads/0.pcd')
# pcd = o3d.io.read_point_cloud(file)
# o3d.io.write_point_cloud('shabalaba.ply', pcd)
# frame = '1'
# folder_save_pcd = 'hammerpcd-re/' # folder to save the pcd files in
#
# ply_path = "data/hammer_upright_ed/" # path to the ply file
#
#plyfile = o3d.io.read_point_cloud(ply_path+frame+'.ply')
#
# o3d.io.write_point_cloud(ply_path+folder_save_pcd+frame+'.pcd', plyfile)

#############################multiple############################################################
#
# import os
# from pathlib import Path
# import open3d as o3d
#
# pathtofolder = os.path.join(Path.home(), 'libroyale-4.24.0.1201-LINUX-x86-64Bit/python/dataset_final_iA/')
# # print(pathtofolder)
# pathtosave = os.path.join(Path.home(), 'libroyale-4.24.0.1201-LINUX-x86-64Bit/python/data/')
#
# for i in range(len(os.listdir(pathtofolder))):
#     pcd = o3d.io.read_point_cloud(pathtofolder+str(i)+".ply")
#     o3d.io.write_point_cloud(pathtosave+str(i)+".pcd", pcd)

#################################################################################################
########################FPS all files in folder##################################################

# import os
#
# path = os.path.abspath('data')
# object = "hammer_upright_done\hammerpcd-re"
# ob_path = os.path.join(path, object) # path to edi.pcd files
#
# for f in os.listdir(ob_path):
#     pcd = o3d.io.read_point_cloud(ob_path+'/'+f)
#     pcd_down = pcd.farthest_point_down_sample(4096)
#     o3d.io.write_point_cloud(ob_path+'/'+'4096_'+str(f), pcd_down)
#

#################################################################################################
######################################HDF5 maker#################################################

# import h5py
# import os, os.path
# import numpy as np
# from plyfile import PlyData, PlyElement
# import glob
# import open3d as o3d
# from pathlib import Path

# NUM_FRAMES = 60

# data = np.zeros((NUM_FRAMES, 1024, 3), dtype=np.float32)
# label = np.zeros((NUM_FRAMES, 1), dtype=np.uint8)
# pid = np.zeros((NUM_FRAMES, 1024), dtype=np.uint8)
# # label = np.zeros((NUM_FRAMES, 4096), dtype=np.uint8)



# home = str(Path.home())
# f = h5py.File(home+'/Downloads/real_data.h5', 'w')
# labeled_data_dir = home + '/Downloads/ply_down_labeled/'
# # #print(labeled_data_dir)
# i = -1
# #
# # #print(glob.glob(labeled_data_dir + '*.ply'))
# #
# #
# #
# for file in glob.iglob(labeled_data_dir + '*.ply'):
#     if (i == NUM_FRAMES - 1):
#         break;
#     i = i + 1
#     xyz_label_ply = PlyData.read(file)

#     for j in range(0, 1024):
#         data[i, j] = [xyz_label_ply['vertex']['x'][j], xyz_label_ply['vertex']['y'][j], xyz_label_ply['vertex']['z'][j]]
#         label[i] = i
#         pid[i, j] = xyz_label_ply['vertex']['scalar_label'][j] # label to pid
#         #print(data[i,j])

#     #print(data.shape)

# f.create_dataset('data', data=data)
# f.create_dataset('label', data=label)
# f.create_dataset('pid', data=pid)
#

############################## change numbering 0->10 1->11 ######################################

# import os
#
# path = os.path.abspath('dataset_final_iA/tape/')
#
#
# for i in range(len(os.listdir(path))):
#     os.rename(path+'/'+str(i)+'.ply', path+'/'+str(i+50)+'.ply')
#
#

################################# RANSACing ground plane #################################################
# import subprocess
# import time
#
#
# wait_time = 3
# for i in range(len(os.listdir('data/'))):
#     os.rename('data/'+str(i)+".pcd", 'data/royale.pcd')
#     time.sleep(wait_time)
#     subprocess.call(['./pcd_write'])
#     print("now waiting for %d second" % (wait_time))
#     time.sleep(wait_time)
#     os.rename("data/pcdGOAT.pcd", "data/"+str(i)+".pcd")
#     os.remove("data/royale.pcd")


########################### FPS #########################################################################

# import os
# from pathlib import Path
#
# path = os.path.join(Path.home(), 'libroyale-4.24.0.1201-LINUX-x86-64Bit/python/data/')
# path_tosave = os.path.join(Path.home(), 'libroyale-4.24.0.1201-LINUX-x86-64Bit/python/data/pcd_down/')
#
# for i in range(len(os.listdir(path))):
#     pcd = o3d.io.read_point_cloud(path+str(i)+'.pcd')
#     pcd_down = pcd.farthest_point_down_sample(1024)
#     o3d.io.write_point_cloud(path_tosave + str(i)+'.pcd', pcd_down)


######################################## add points ####################################################
#
# pcd = o3d.io.read_point_cloud("data/48.pcd")
# points = np.asarray(pcd.points)
#
#
# for i in range(28):
#     a = int(np.random.uniform(0,9,1))
#     point = [0.0310222,  0.0841162, float('0.4'+str(a))]
#     # print(point)
#     pcd.points.append(point)
# print(len(pcd.points))
# o3d.io.write_point_cloud("data/49.pcd",pcd)


######################################## example ####################################################

# import open3d as o3d
# from pathlib import Path
# import subprocess
# import os
# import numpy as np

# path = os.path.join(Path.home(),'Downloads')

# pcd = o3d.io.read_point_cloud(path+'/example.pcd')
# points = np.asarray(pcd.points)

# # o3d.visualization.draw_geometries([pcd])

# pcd = pcd.select_by_index(np.where((points[:, 1] < 0.5) & (points[:, 1] > 0.05)
#                                        & (points[:, 0] < 0.2) & (points[:, 0] > -0.2)
#                                        & (points[:, 2] < 0.6) & (points[:, 2] > 0.3)
#                                        )[-1])

# o3d.io.write_point_cloud(path+'/../detection_pipeline/data/pcd_extra.pcd', pcd)
# pcd_extra = o3d.io.read_point_cloud(path+'/../detection_pipeline/data/pcd_extra.pcd')
# o3d.visualization.draw_geometries([pcd_extra])

# os.rename('data/pcd_extra.pcd', 'data/royale.pcd')
# subprocess.call(['./pcd_write'])
# pcdGOAT = o3d.io.read_point_cloud(path+'/../detection_pipeline/data/pcdGOAT.pcd')
# o3d.visualization.draw_geometries([pcdGOAT])

####################################### test ############################################################

# import open3d as o3d
# from pathlib import Path
# import subprocess
# import os
# import numpy as np

# path = os.path.join(Path.home(),'Downloads/../detection_pipeline/data/')
# print(path)

# pcd = o3d.io.read_point_cloud(path+'/5.pcd')
# points = np.asarray(pcd.points)

# pcd = pcd.select_by_index(np.where((points[:, 1] < 0.5) & (points[:, 1] > 0.05)
#                                        & (points[:, 0] < 0.2) & (points[:, 0] > -0.2)
#                                        & (points[:, 2] < 0.6) & (points[:, 2] > 0.3)
#                                        )[-1])

# # o3d.visualization.draw_geometries([pcd])

# os.rename('data/5.pcd', 'data/royale.pcd')
# subprocess.call(['./pcd_write'])
# pcdGOAT = o3d.io.read_point_cloud(path+'/pcdGOAT.pcd')
# o3d.visualization.draw_geometries([pcdGOAT])

########################################## algo ##########################################################

import open3d as o3d 
import numpy as np
