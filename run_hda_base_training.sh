#!/bin/bash
#SBATCH  --output=sbatch_log/%j.out
#SBATCH  --gres=gpu:4
#SBATCH  --mem=30G

source /scratch_net/biwidl13/yihshe/conda/etc/profile.d/conda.sh
conda activate pytcu10

python3 -m tools.train_net --num-gpus 4 \
        --config-file configs/COCO-detection/faster_rcnn_R_101_FPN_hda_base.yaml 

# python3 -m tools.test_net --num-gpus 2\
#         --config-file configs/COCO-detection/faster_rcnn_R_101_FPN_hda_base.yaml \
#         --eval-only
