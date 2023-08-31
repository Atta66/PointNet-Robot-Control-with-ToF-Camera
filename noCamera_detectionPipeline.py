import os
import subprocess
import numpy as np
import open3d as o3d
import time
import model
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
        o3d.io.write_point_cloud(self.out_ply, self.pcd_down)
        o3d.visualization.draw_geometries([self.pcd_down])
        # print(len(self.pcd_down.points))
        return self.pcd_down


if __name__ == '__main__':
    
    # number of frames to be recorded
    frames_toExtract = '1'

    # # export rrf file
    out_rrf = '1'
    
    # # output ply file ==== it is the same as out_rrf + .ply
    # out_ply = '0.ply'

    # pp_pc = data_preprocess(frames_toExtract, out_rrf)
    # pp_pc.record_frames()
    # pp_pc.convert_to_ply()
    # pp_pc.distance_filter()
    # pp_pc.RANSAC()
    # # pp_pc.fps_downsample()

    # pcd = pp_pc.fps_downsample()


    pcd = o3d.io.read_point_cloud('data/real_data/t1.pcd')
    pcd = pcd.farthest_point_down_sample(1024)
    points = np.asarray(pcd.points)



    sampled_point_cloud = points

    norm_point_cloud = sampled_point_cloud - np.mean(sampled_point_cloud, axis=0)
    norm_point_cloud /= np.max(np.linalg.norm(norm_point_cloud, axis=1))

    pcd_unseen = norm_point_cloud
    pcd_unseen = pcd_unseen.reshape(1,1024,3)

    segmentation_model = run_experiment()

    val_predictions = segmentation_model.predict(pcd_unseen)
   
    pcd_unseen = pcd_unseen.reshape(1024,3)

    classes = ['POI_box','POI_tape','nPOI_tape','POI_extension','nPOI_extension','POI_hammer',
             'nPOI_hammer', 'POI_pliers', 'nPOI_pliers','POI_cutter', 'nPOI_cutter']
    
    label_cloud = val_predictions.reshape(1024,11)
    babu = []
    for pred in label_cloud:
        # print(np.argmax(pred))
        if classes[np.argmax(pred)] not in babu:
         babu.append(classes[np.argmax(pred)])
    print(babu)


    




    label_map = classes + ["none"]

    label_cloud = val_predictions.reshape(1024,11)

    visualize_data(pcd_unseen, [label_map[np.argmax(label)] for label in label_cloud])

    # Plotting with predicted labels.
    # visualize_single_point_cloud(pcd_unseen, val_predictions) 