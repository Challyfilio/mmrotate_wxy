#!/bin/bash
echo "* * * * * * * * * * * * * * * * * * * * *"
echo "Hello, what a nice day! You are so great!"
echo "* * * * * * * * * * * * * * * * * * * * *"
# trainval
cd ..
#METHOD=oriented_rcnn_r50_fpn_ce_dota_ss_traintest0.8
METHOD=$1
STATE=$2
#BACKBONE=$2
#NECK=$3
#LOSS=$4
#DATASET=$5
echo ${METHOD}

CONFIG=oriented_rcnn/oriented_rcnn_r50_fpn_fp16_1x_dota_le90.py
WORK_DIR=work_dir/${METHOD}
TEST_DIR=test_dir/${METHOD}/Task1_results

if(($STATE==0 || $STATE==2)); then
#  python tools/train.py \
#          configs/${CONFIG} \
#          --no-validate \
#          --work-dir ${WORK_DIR}

  python tools/train.py \
          configs/${CONFIG} \
          --work-dir ${WORK_DIR}
fi

if(($STATE==1 || $STATE==2)); then
  if [ -d "$TEST_DIR" ]; then
      echo "Results are available in ${TEST_DIR}. Skip this job"
  else
    python tools/test.py \
           configs/${CONFIG} \
           ${WORK_DIR}/latest.pth \
           --format-only \
           --eval-options \
           submission_dir=${TEST_DIR}
  fi
fi

echo "Finish!"