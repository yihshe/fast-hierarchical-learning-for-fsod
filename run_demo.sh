#!/bin/bash
#SBATCH  --output=sbatch_log/%j.out
#SBATCH  --gres=gpu:1 
#SBATCH  --mem=30G
source /scratch_net/biwidl13/yihshe/conda/etc/profile.d/conda.sh
conda activate pytcu10
# TFA
# python3 -m demo.demo \
#         --config-file configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_10shot.yaml \
#         --input datasets/coco/train2014/COCO_train2014_000000557660.jpg \
#         --output checkpoints_temp/figs/Nov15_visual/TFA/base/laptop \
#         --opts MODEL.WEIGHTS checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_10shot/model_final.pth

python3 -m run_small_tasks9_demo_visual

