#!/bin/bash
#SBATCH  --output=sbatch_log/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=30G
#SBATCH  --constraint='titan_xp'

source /scratch_net/biwidl13/yihshe/conda/etc/profile.d/conda.sh
conda activate pytcu10

# Initialize the novel weights
# python3 -m tools.ckpt_surgery \
#         --src1 checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_base/model_final.pth \
#         --method remove 

# python3 -m tools.train_net --num-gpus 1 \
#         --config-file configs/COCO-detection/faster_rcnn_R_101_FPN_ft_novel_1shot.yaml \
#         --opts MODEL.WEIGHTS checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_base/model_reset_remove.pth

# python3 -m tools.ckpt_surgery \
#         --src1 checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_base/model_final.pth \
#         --src2 checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel_1shot/model_final.pth \
#         --method combine \
#         --coco \
#         --save-dir checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel_1shot_combine

# Random initialization
# python3 -m tools.ckpt_surgery \
#         --src1 checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1/model_final.pth \
#         --method randinit \
#         --save-dir checkpoints_temp/voc/faster_rcnn/faster_rcnn_R_101_FPN_all1

# python3 -m tools.train_net --num-gpus 2 \
#         --config-file configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_1shot.yaml \
#         --opts MODEL.WEIGHTS checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel_1shot_combine/model_reset_combine.pth

# for running SGD after feature extraction, which can share same config file with CG
python3 -m tools.train_net_modified --num-gpus 1 \
        --config-file configs/COCO-detection_CG/faster_rcnn_R_101_FPN_ft_all_1shot.yaml \
        --opts MODEL.WEIGHTS checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel_1shot_combine/model_reset_combine.pth