import os
import subprocess
import numpy as np
import open3d as o3d
import time
from collections import Counter
from model import *



class data_preprocess():

    def __init__(self, frames_toExtract, out_rrf):
        self.frames_toExtract = frames_toExtract
        self.out_rrf = out_rrf
        self.out_ply = str(out_rrf)+'.ply'
        self.proc = None

    def record_frames(self):
        os.system('./sampleRecordRRF '+str(self.frames_toExtract)+' '+str(self.out_rrf))

    def convert_to_ply(self):
        subprocess.call(['./sampleExportPLY', str(self.out_rrf)])

    def distance_filter(self): # instead of radius traditional filtering
        self.pcd = o3d.io.read_point_cloud(self.out_ply)
        self.points = np.asarray(self.pcd.points)

        self.pcd_filtered = self.pcd.select_by_index(np.where((self.points[:, 1] < 0.5) & (self.points[:, 1] > 0.05)
                                       & (self.points[:, 0] < 0.2) & (self.points[:, 0] > -0.2)
                                       & (self.points[:, 2] < 0.6) & (self.points[:, 2] > 0.3)
                                       )[-1])

        o3d.io.write_point_cloud("data/"+out_rrf+".pcd", self.pcd_filtered)
        # o3d.visualization.draw_geometries([self.pcd_filtered])

    def RANSAC(self):
        
        wait_time = 3
        os.rename("data/"+self.out_rrf+".pcd", 'data/royale.pcd')
        
        # time.sleep(wait_time)
        subprocess.call(['./pcd_write'])
        print("now waiting for %d second" % (wait_time))
        time.sleep(wait_time)
        os.rename("data/pcdGOAT.pcd", "data/"+self.out_rrf+".pcd")
        os.remove("data/royale.pcd")

    def fps_downsample(self):
        self.pcd_filtered = o3d.io.read_point_cloud("data/"+self.out_rrf+".pcd")
        self.pcd_down = self.pcd_filtered.farthest_point_down_sample(1024)
        o3d.io.write_point_cloud("data/"+self.out_rrf+".pcd", self.pcd_down)
        # o3d.visualization.draw_geometries([self.pcd_down])
        # print(len(self.pcd_down.points))
        return self.pcd_down



#def control():




def dist_x(classes, label_cloud):
    
    pred_poi_x = 0
    pred_poi_x_sum = 0
    pred_npoi_x = 0
    pred_npoi_x_sum = 0

    for i in range(1024):
        if classes[np.argmax(label_cloud[i])] == "POI_pliers" and pred_poi_x != 100:
            pred_poi_x += 1
            pred_poi_x_sum += points[i][0]
        if classes[np.argmax(label_cloud[i])] == "nPOI_pliers" and pred_npoi_x != 100:
            pred_npoi_x += 1
            pred_npoi_x_sum += points[i][0]
            
    print("pred_poi x = %d, pred_npoi x= %d, pred_difference x= %f" % (pred_poi_x, pred_npoi_x, pred_poi_x_sum - pred_npoi_x_sum))

    return pred_poi_x, pred_npoi_x, pred_poi_x_sum - pred_npoi_x_sum

def dist_z(classes, label_cloud):

    pred_poi_z = 0
    pred_poi_z_sum = 0
    pred_npoi_z = 0
    pred_npoi_z_sum = 0

    for i in range(1024):
        if classes[np.argmax(label_cloud[i])] == "POI_pliers" and pred_poi_z != 100:
            pred_poi_z += 1
            pred_poi_z_sum += points[i][2]
        if classes[np.argmax(label_cloud[i])] == "nPOI_pliers" and pred_npoi_z != 100:
            pred_npoi_z += 1
            pred_npoi_z_sum += points[i][2]
            
    print("pred_poi z = %d, pred_npoi z= %d, pred_difference z= %f" % (pred_poi_z, pred_npoi_z, pred_poi_z_sum - pred_npoi_z_sum))

    return pred_poi_z, pred_npoi_z, pred_poi_z_sum - pred_npoi_z_sum




if __name__ == '__main__':

    # segmentation model
    segmentation_model = run_experiment()

    # classes
    classes = ['POI_box','POI_tape','nPOI_tape','POI_extension','nPOI_extension','POI_drill',
             'nPOI_drill', 'POI_pliers', 'nPOI_pliers','POI_cutter', 'nPOI_cutter']

    # number of times same PC segmented
    num_times = 0

    # if robot has moved before
    moved = 0

    while True:

        input("press enter to start pipeline again...")

        # number of frames to be recorded
        frames_toExtract = '1'

        # # export rrf file
        out_rrf = '1'
    
        # output ply file ==== it is the same as out_rrf + .ply
        # out_ply = '0.ply'

        # start recording and preprocessing of PC
        pp_pc = data_preprocess(frames_toExtract, out_rrf)
        pp_pc.record_frames()
        pp_pc.convert_to_ply()
        pp_pc.distance_filter()
        pp_pc.RANSAC()

        # preprocessed PC for detection
        pp_forDetection = pp_pc.fps_downsample()
        points = np.asarray(pp_forDetection.points)

        # normalize PC
        norm_point_cloud = points - np.mean(points, axis=0)
        norm_point_cloud /= np.max(np.linalg.norm(norm_point_cloud, axis=1))

        # reshape for detection
        pcd_reshaped = norm_point_cloud.reshape(1,1024,3)

        # segment PC
        seg_PC = segmentation_model.predict(pcd_reshaped)

        # seen classes in segmentation result
        label_cloud = seg_PC.reshape(1024,11)
        seen_classes = []
        for pred in label_cloud:
            if classes[np.argmax(pred)] not in seen_classes:
                seen_classes.append(classes[np.argmax(pred)])
        print(seen_classes)


        # distance measurement between points
        p_x, np_x, diff_x = dist_x(classes, label_cloud)
        p_z, np_z, diff_z = dist_z(classes, label_cloud)

        
        # for accuracy each PC passed two times for segmentation
        if num_times == 0:
            segmented = []
            for obj in seen_classes:
                segmented.append(obj)
            num_times = 1

        elif num_times == 1 and Counter(map(str, seen_classes)) == Counter(map(str, segmented)):
            
            if seen_classes == ['POI_drill', 'nPOI_drill'] or seen_classes == ['nPOI_drill', 'POI_drill']:

                # S
                if 0 <= diff_z <= 3 and  5 <= diff_x <= 9:
                    print("go right")
                    moved = 1
                    num_times = 0
                # DC
                elif -6 <= diff_z <= 3 and -7.5 <= diff_x <= 3.5:
                    print("go up right")
                    moved = 1
                    num_times = 0
                # VC
                elif -14 <= diff_z <= -11 and 0 <= diff_x <= 6:
                    print("move up")
                    moved = 1
                    num_times = 0
                # S
                elif -0.5 <= diff_z <= 3 and -13 <= diff_x <= -5:
                    print("move left")
                    moved = 1
                    num_times = 0
                # DF
                elif -30 <= diff_z <= -12 and -10 <= diff_x <= 0:
                    print("move left")
                    moved = 1
                    num_times = 0
                # DF
                elif -50 <= diff_z <= -30 and -1 <= diff_x <= 4:
                    print("move right")
                    moved = 1
                    num_times = 0 
            # VF
            elif seen_classes == ['nPOI_drill'] and -47 <= diff_z <= -35 and 0 <= diff_x <= 6:
                print("move right")
                moved = 1
                num_times = 0


        elif num_times == 1 and moved == 0 and Counter(map(str, seen_classes)) != Counter(map(str, segmented)):
            print("move right")
            num_times = 0

        elif num_times == 1 and moved == 1 and Counter(map(str, seen_classes)) != Counter(map(str, segmented)):
            print("same direction as before")
            num_times = 0




        # visualization
        # label_map = classes + ["none"]
        # visualize_data(norm_point_cloud, [label_map[np.argmax(label)] for label in label_cloud])



