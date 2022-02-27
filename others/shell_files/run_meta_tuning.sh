#!/bin/bash
#SBATCH  --output=sbatch_log/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=30G
#SBATCH  --constraint='titan_xp|geforce_gtx_titan_x'

source /scratch_net/biwidl13/yihshe/conda/etc/profile.d/conda.sh
conda activate pytcu10

# python3 -m tools.meta_tune_net --num-gpus 1 \
#         --config-file configs/COCO-detection_CG_meta/faster_rcnn_R_101_FPN_ft_all_1shot.yaml \
#         --opts META_PARAMS.MODEL_WEIGHTS checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_base/model_final.pth
python3 -m tools.meta_tune_net --num-gpus 1 \
        --config-file configs/COCO-detection_CG_meta/faster_rcnn_R_101_FPN_ft_novel_1shot.yaml \
        --opts META_PARAMS.MODEL_WEIGHTS checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_base/model_reset_remove.pth