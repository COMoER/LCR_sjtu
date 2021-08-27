#!/bin/bash
root_path=~/DATA/Workspace/zy_stock/LCR_sjtu # change to your root path
livox_path=/home/sjtu/DATA/Workspace/zy_stock/yolov5_4.0_radar/lidar_area
cd $root_path 
gnome-terminal --geometry 60x20+10+10 -- bash $livox_path/start.sh
gnome-terminal --geometry 60x20+10+10 -- bash main_l.sh
