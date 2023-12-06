#!/bin/bash
img_list='P0500 P0628 P0680 P0681 P1511 P2645'

MODE=$1

cd ..

for IMG in ${img_list}; do
  python demo/image_demo.py \
   demo/${IMG}.png \
   configs/oriented_rcnn/ORDF50.py \
   work_dir/oriented_rcnn_r50_dcn_pafpnwo_adapter_ce_dota_${MODE}_traintest0.8/latest.pth \
   --out-file ${IMG}_${MODE}.jpg
done