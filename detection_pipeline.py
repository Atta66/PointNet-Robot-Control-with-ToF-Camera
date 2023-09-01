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
        self.points = np.asarray(self.pcd_filtered.points)
        if len(self.points) < 1024:
            return self.pcd_filtered

        else:
            self.pcd_down = self.pcd_filtered.farthest_point_down_sample(1024)
            o3d.io.write_point_cloud("data/"+self.out_rrf+".pcd", self.pcd_down)
            # o3d.visualization.draw_geometries([self.pcd_down])
            # print(len(self.pcd_down.points))
            return self.pcd_down



#def control():




def dist_x(classes, label_cloud):
    
    # drill
    pred_poi_drill_x = 0
    pred_poi_drill_x_sum = 0
    pred_npoi_drill_x = 0
    pred_npoi_drill_x_sum = 0

    # box
    pred_poi_box_x = 0
    pred_poi_box_x_sum = 0

    # tape
    pred_poi_tape_x = 0
    pred_poi_tape_x_sum = 0
    pred_npoi_tape_x = 0
    pred_npoi_tape_x_sum = 0

    # cutter
    pred_poi_cutter_x = 0
    pred_poi_cutter_x_sum = 0
    pred_npoi_cutter_x = 0
    pred_npoi_cutter_x_sum = 0

    # extension
    pred_poi_extension_x = 0
    pred_poi_extension_x_sum = 0
    pred_npoi_extension_x = 0
    pred_npoi_extension_x_sum = 0

    # pliers
    pred_poi_pliers_x = 0
    pred_poi_pliers_x_sum = 0
    pred_npoi_pliers_x = 0
    pred_npoi_pliers_x_sum = 0

    for i in range(1024):
        if classes[np.argmax(label_cloud[i])] == "POI_drill" and pred_poi_drill_x != 100:
            pred_poi_drill_x += 1
            pred_poi_drill_x_sum += points[i][0]
        if classes[np.argmax(label_cloud[i])] == "nPOI_drill" and pred_npoi_drill_x != 100:
            pred_npoi_drill_x += 1
            pred_npoi_drill_x_sum += points[i][0]
        if classes[np.argmax(label_cloud[i])] == "POI_box" and pred_poi_box_x != 100:
            pred_poi_box_x += 1
            pred_poi_box_x_sum += points[i][0]
        

        if classes[np.argmax(label_cloud[i])] == "POI_tape" and pred_poi_tape_x != 100:
            pred_poi_tape_x += 1
            pred_poi_tape_x_sum += points[i][0]
        if classes[np.argmax(label_cloud[i])] == "nPOI_tape" and pred_npoi_tape_x != 100:
            pred_npoi_tape_x += 1
            pred_npoi_tape_x_sum += points[i][0]
        if classes[np.argmax(label_cloud[i])] == "POI_cutter" and pred_poi_cutter_x != 100:
            pred_poi_cutter_x += 1
            pred_poi_cutter_x_sum += points[i][0]
        if classes[np.argmax(label_cloud[i])] == "nPOI_cutter" and pred_npoi_cutter_x != 100:
            pred_npoi_cutter_x += 1
            pred_npoi_cutter_x_sum += points[i][0]
        if classes[np.argmax(label_cloud[i])] == "POI_extension" and pred_poi_extension_x != 100:
            pred_poi_extension_x += 1
            pred_poi_extension_x_sum += points[i][0]
        if classes[np.argmax(label_cloud[i])] == "nPOI_extension" and pred_npoi_extension_x != 100:
            pred_npoi_extension_x += 1
            pred_npoi_extension_x_sum += points[i][0]
        if classes[np.argmax(label_cloud[i])] == "POI_pliers" and pred_poi_pliers_x != 100:
            pred_poi_pliers_x += 1
            pred_poi_pliers_x_sum += points[i][0]
        if classes[np.argmax(label_cloud[i])] == "nPOI_pliers" and pred_npoi_pliers_x != 100:
            pred_npoi_pliers_x += 1
            pred_npoi_pliers_x_sum += points[i][0]
            
    print("DRILL: pred_poi x = %d, pred_npoi x= %d, pred_difference x= %f" % (pred_poi_drill_x, pred_npoi_drill_x, pred_poi_drill_x_sum - pred_npoi_drill_x_sum))
    print("BOX: pred_poi x = %d, pred_difference x= %f" % (pred_poi_box_x, pred_poi_box_x_sum))
    print("TAPE: pred_poi x = %d, pred_npoi x= %d, pred_difference x= %f" % (pred_poi_tape_x, pred_npoi_tape_x, pred_poi_tape_x_sum - pred_npoi_tape_x_sum))
    print("CUTTER: pred_poi x = %d, pred_npoi x= %d, pred_difference x= %f" % (pred_poi_cutter_x, pred_npoi_cutter_x, pred_poi_cutter_x_sum - pred_npoi_cutter_x_sum))
    print("EXTENSION: pred_poi x = %d, pred_npoi x= %d, pred_difference x= %f" % (pred_poi_extension_x, pred_npoi_extension_x, pred_poi_extension_x_sum - pred_npoi_extension_x_sum))
    print("PLIERS: pred_poi x = %d, pred_npoi x= %d, pred_difference x= %f" % (pred_poi_pliers_x, pred_npoi_pliers_x, pred_poi_pliers_x_sum - pred_npoi_pliers_x_sum))

    diff_drill = pred_poi_drill_x_sum - pred_npoi_drill_x_sum
    diff_box = pred_poi_box_x
    diff_tape = pred_poi_tape_x_sum - pred_npoi_tape_x_sum
    diff_cutter = pred_poi_cutter_x_sum - pred_npoi_cutter_x_sum
    diff_extension = pred_poi_extension_x_sum - pred_npoi_extension_x_sum
    diff_pliers = pred_poi_pliers_x_sum - pred_npoi_pliers_x_sum
    

    return diff_drill, diff_box, diff_tape, diff_cutter, diff_extension, diff_pliers

def dist_z(classes, label_cloud):

    # drill
    pred_poi_drill_z = 0
    pred_poi_drill_z_sum = 0
    pred_npoi_drill_z = 0
    pred_npoi_drill_z_sum = 0

    # box
    pred_poi_box_z = 0
    pred_poi_box_z_sum = 0

    # tape
    pred_poi_tape_z = 0
    pred_poi_tape_z_sum = 0
    pred_npoi_tape_z = 0
    pred_npoi_tape_z_sum = 0

    # cutter
    pred_poi_cutter_z = 0
    pred_poi_cutter_z_sum = 0
    pred_npoi_cutter_z = 0
    pred_npoi_cutter_z_sum = 0

    # extension
    pred_poi_extension_z = 0
    pred_poi_extension_z_sum = 0
    pred_npoi_extension_z = 0
    pred_npoi_extension_z_sum = 0

    # pliers
    pred_poi_pliers_z = 0
    pred_poi_pliers_z_sum = 0
    pred_npoi_pliers_z = 0
    pred_npoi_pliers_z_sum = 0

    for i in range(1024):
        if classes[np.argmax(label_cloud[i])] == "POI_drill" and pred_poi_drill_z != 100:
            pred_poi_drill_z += 1
            pred_poi_drill_z_sum += points[i][2]
        if classes[np.argmax(label_cloud[i])] == "nPOI_drill" and pred_npoi_drill_z != 100:
            pred_npoi_drill_z += 1
            pred_npoi_drill_z_sum += points[i][2]
        if classes[np.argmax(label_cloud[i])] == "POI_box" and pred_poi_box_z != 100:
            pred_poi_box_z += 1
            pred_poi_box_z_sum += points[i][2]
        

        if classes[np.argmax(label_cloud[i])] == "POI_tape" and pred_poi_tape_z != 100:
            pred_poi_tape_z += 1
            pred_poi_tape_z_sum += points[i][2]
        if classes[np.argmax(label_cloud[i])] == "nPOI_tape" and pred_npoi_tape_z != 100:
            pred_npoi_tape_z += 1
            pred_npoi_tape_z_sum += points[i][2]
        if classes[np.argmax(label_cloud[i])] == "POI_cutter" and pred_poi_cutter_z != 100:
            pred_poi_cutter_z += 1
            pred_poi_cutter_z_sum += points[i][2]
        if classes[np.argmax(label_cloud[i])] == "nPOI_cutter" and pred_npoi_cutter_z != 100:
            pred_npoi_cutter_z += 1
            pred_npoi_cutter_z_sum += points[i][2]
        if classes[np.argmax(label_cloud[i])] == "POI_extension" and pred_poi_extension_z != 100:
            pred_poi_extension_z += 1
            pred_poi_extension_z_sum += points[i][2]
        if classes[np.argmax(label_cloud[i])] == "nPOI_extension" and pred_npoi_extension_z != 100:
            pred_npoi_extension_z += 1
            pred_npoi_extension_z_sum += points[i][2]
        if classes[np.argmax(label_cloud[i])] == "POI_pliers" and pred_poi_pliers_z != 100:
            pred_poi_pliers_z += 1
            pred_poi_pliers_z_sum += points[i][2]
        if classes[np.argmax(label_cloud[i])] == "nPOI_pliers" and pred_npoi_pliers_z != 100:
            pred_npoi_pliers_z += 1
            pred_npoi_pliers_z_sum += points[i][2]
            
    print("DRILL: pred_poi z = %d, pred_npoi z= %d, pred_difference z= %f" % (pred_poi_drill_z, pred_npoi_drill_z, pred_poi_drill_z_sum - pred_npoi_drill_z_sum))
    print("BOX: pred_poi z = %d, pred_difference z= %f" % (pred_poi_box_z, pred_poi_box_z_sum))
    print("TAPE: pred_poi z = %d, pred_npoi z= %d, pred_difference z= %f" % (pred_poi_tape_z, pred_npoi_tape_z, pred_poi_tape_z_sum - pred_npoi_tape_z_sum))
    print("CUTTER: pred_poi z = %d, pred_npoi z= %d, pred_difference z= %f" % (pred_poi_cutter_z, pred_npoi_cutter_z, pred_poi_cutter_z_sum - pred_npoi_cutter_z_sum))
    print("EXTENSION: pred_poi z = %d, pred_npoi z= %d, pred_difference z= %f" % (pred_poi_extension_z, pred_npoi_extension_z, pred_poi_extension_z_sum - pred_npoi_extension_z_sum))
    print("PLIERS: pred_poi z = %d, pred_npoi z= %d, pred_difference z= %f" % (pred_poi_pliers_z, pred_npoi_pliers_z, pred_poi_pliers_z_sum - pred_npoi_pliers_z_sum))

    diff_drill = pred_poi_drill_z_sum - pred_npoi_drill_z_sum
    diff_box = pred_poi_box_z
    diff_tape = pred_poi_tape_z_sum - pred_npoi_tape_z_sum
    diff_cutter = pred_poi_cutter_z_sum - pred_npoi_cutter_z_sum
    diff_extension = pred_poi_extension_z_sum - pred_npoi_extension_z_sum
    diff_pliers = pred_poi_pliers_z_sum - pred_npoi_pliers_z_sum
    

    return diff_drill, diff_box, diff_tape, diff_cutter, diff_extension, diff_pliers




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

        input("press enter to start pipeline...")

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


        # move 1/8th of the circle
        if len(points) < 1024:
            print("go double right")
            continue

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
        diff_drill_x, diff_box_x, diff_tape_x, diff_cutter_x, diff_extension_x, diff_pliers_x = dist_x(classes, label_cloud)
        diff_drill_z, diff_box_z, diff_tape_z, diff_cutter_z, diff_extension_z, diff_pliers_z = dist_z(classes, label_cloud)

        
        # for accuracy each PC passed two times for segmentation
        if num_times == 0:
            segmented = []
            for obj in seen_classes:
                segmented.append(obj)
            num_times = 1

        elif num_times == 1 and Counter(map(str, seen_classes)) == Counter(map(str, segmented)):
            
            if seen_classes == ['POI_drill', 'nPOI_drill'] or seen_classes == ['nPOI_drill', 'POI_drill']:

                # S
                if 0 <= diff_drill_z <= 3 and  5 <= diff_drill_x <= 9:
                    print("go right")
                    moved = 1
                    num_times = 0
                # DC
                elif -6 <= diff_drill_z <= 3 and -7.5 <= diff_drill_x <= 3.5:
                    print("go up right")
                    moved = 1
                    num_times = 0
                # VC
                elif -14 <= diff_drill_z <= -11 and 0 <= diff_drill_x <= 6:
                    print("move up")
                    moved = 1
                    num_times = 0
                # S
                elif -0.5 <= diff_drill_z <= 5 and -13 <= diff_drill_x <= -5:
                    print("move left")
                    moved = 1
                    num_times = 0
                # DF
                elif -30 <= diff_drill_z <= -12 and -10 <= diff_drill_x <= 0:
                    print("move left")
                    moved = 1
                    num_times = 0
                # DF
                elif -50 <= diff_drill_z <= -30 and -1 <= diff_drill_x <= 4:
                    print("move right")
                    moved = 1
                    num_times = 0 
                else:
                    print("move left")
                    moved = 1
                    num_times = 0
            # VF
            elif seen_classes == ['nPOI_drill'] and -47 <= diff_drill_z <= -35 and 0 <= diff_drill_x <= 6:
                print("move right or left")
                moved = 1
                num_times = 0

            elif seen_classes == ['POI_cutter', 'nPOI_cutter'] or seen_classes == ['nPOI_cutter', 'POI_cutter']:

                # S
                if -1 <= diff_cutter_z <= 2 and  7 <= diff_cutter_x <= 12:
                    print("go right")
                    moved = 1
                    num_times = 0
                # DC
                elif 0 <= diff_cutter_z <= 4.4 and -7.5 <= diff_cutter_x <= -1:
                    print("go up right")
                    moved = 1
                    num_times = 0
                # VC
                elif -45 <= diff_cutter_z <= -35 and 0 <= diff_cutter_x <= 6:
                    print("move up")
                    moved = 1
                    num_times = 0

                # S
                elif -25 <= diff_cutter_z <= 0 and -3 <= diff_cutter_x <= 8:
                    print("move left")
                    moved = 1
                    num_times = 0
                # VF
                elif 0 <= diff_cutter_z <= 8 and -2 <= diff_cutter_x <= 2:
                    print("move left ya right")
                    moved = 1
                    num_times = 0
                # DF
                elif -2 <= diff_cutter_z <= 9 and 3 <= diff_cutter_x <= 10:
                    print("move right")
                    moved = 1
                    num_times = 0
                else:
                    print("move left")
                    moved = 1
                    num_times = 0

            # VC Cutter
            elif seen_classes == ['POI_cutter'] and 35 <= diff_cutter_z <= 52 and -5 <= diff_cutter_x <= 0:
                    print("move up")
                    moved = 1
                    num_times = 0


            elif seen_classes == ['nPOI_extension']:
                # VF
                if -60 <= diff_cutter_z <= 50 and  0 <= diff_cutter_x <= 5:
                    print("go right")
                    moved = 1
                    num_times = 0

            elif seen_classes == ['POI_extension']:
                # DF
                # if 50 <= diff_cutter_z <= 65 and  2 <= diff_cutter_x <= 5:
                print("go up")
                moved = 1
                num_times = 0


            elif seen_classes == ['POI_pliers'] or seen_classes == ['POI_pliers', 'POI_extension'] or seen_classes == ['POI_extension', 'POI_pliers'] or seen_classes == ['nPOI_extension', 'POI_pliers'] or seen_classes == ['nPOI_extension', 'POI_pliers']:
                # DC extension
                if 50 <= diff_pliers_z <= 60 and  0 <= diff_pliers_x <= 10:
                    print("go up left")
                    moved = 1
                    num_times = 0


            elif seen_classes == ['POI_extension', 'nPOI_extension'] or seen_classes == ['nPOI_extension', 'POI_extension']:

                # S
                if 10 <= diff_extension_z <= 20 and  -3 <= diff_extension_x <= 3:
                    print("go right")
                    moved = 1
                    num_times = 0
                # DC
                elif 45 <= diff_extension_z <= 55 and -15 <= diff_extension_x <= -5:
                    print("go up right")
                    moved = 1
                    num_times = 0
                elif 4 <= diff_extension_z <= 14 and -10 <= diff_extension_x <= -2:
                    print("go up right")
                    moved = 1
                    num_times = 0

                # VC
                elif -35 <= diff_extension_z <= -15 and -1 <= diff_extension_x <= 8:
                    print("move up")
                    moved = 1
                    num_times = 0

                # DC
                elif 50 <= diff_extension_z <= 60 and 0 <= diff_extension_x <= 5:
                    print("move up left")
                    moved = 1
                    num_times = 0

                # S
                elif -5 <= diff_extension_z <= 5 and -20 <= diff_extension_x <= 0:
                    print("move left")
                    moved = 1
                    num_times = 0

                # DC Cutter
                elif -50 <= diff_extension_z <= -40 and -5 <= diff_extension_x <= 5:
                    print("move up left")
                    moved = 1
                    num_times = 0  



            elif seen_classes == ['POI_pliers', 'nPOI_pliers'] or seen_classes == ['nPOI_pliers', 'POI_pliers']:

                # S
                if 3 <= diff_pliers_z <= 11 and  5 <= diff_pliers_x <= 10:
                    print("go right")
                    moved = 1
                    num_times = 0
                # DC
                elif diff_pliers_z > 25 and -6 <= diff_pliers_x <= 0:
                    print("go up right")
                    moved = 1
                    num_times = 0
                elif -7 <= diff_pliers_z > 5 and 0 <= diff_pliers_x <= 5:
                    print("go up right")
                    moved = 1
                    num_times = 0

                # DF
                elif 1 <= diff_pliers_z <= 5 and -7 <= diff_pliers_x <= -2:
                    print("move left")
                    moved = 1
                    num_times = 0
                # DF
                elif 4 <= diff_pliers_z <= 6 and 4 <= diff_pliers_x <= 6:
                    print("move right")
                    moved = 1
                    num_times = 0 

                elif seen_classes == ['POI_box']:
                    print("move up")
                    moved = 1
                    num_times = 0

            elif seen_classes == ['POI_pliers']:
                print("move up right")
                moved = 1
                num_times = 0

            elif seen_classes == ['POI_tape', 'nPOI_tape'] or seen_classes == ['nPOI_tape', 'POI_tape']:
                print("move up")
                moved = 1
                num_times = 0


        elif num_times == 1 and moved == 0 and Counter(map(str, seen_classes)) != Counter(map(str, segmented)):
            print("move right")
            moved = 1
            num_times = 0

        elif num_times == 1 and moved == 1 and Counter(map(str, seen_classes)) != Counter(map(str, segmented)):
            print("same direction as before")
            num_times = 0




        # visualization
        # label_map = classes + ["none"]
        # visualize_data(norm_point_cloud, [label_map[np.argmax(label)] for label in label_cloud])



