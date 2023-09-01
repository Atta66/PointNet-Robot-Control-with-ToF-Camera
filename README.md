# Movement Control of TB2 using PointNet with a ToF Camera

## Python

`detection_pipeline.py:` Main automated pipeline for segmentation and movement control output

`model.py:` PointNet Model which is imported by `detection_pipeline.py`

## CPP Executables

All the executable are used in `detection_pipeline.py`

`sampleRecordRRF:` for recording RRF

`sampleExportPLY:` RRF to PLY converter

`pcd_write:` Ground plane removal algorithm (RANSAC)

## Other

`1:` RRF output

`1.ply:` Result from RRF to PLY conversion

`checkpoint:` Contains the segmentation ckpt with 73 % accuracy on training data

`data:` pcd files for testing purposes



## Video demonstration

This [video](https://drive.google.com/file/d/1Y32wGbo0B1l3KgflX6xStp7I16IVYUCs/view?usp=sharing) shows TB2 navigating around a drill machine to reach the PoI.
