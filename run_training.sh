#!/bin/bash
#SBATCH  --output=sbatch_log/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=30G
#SBATCH  --constraint='titan_xp'

source /scratch_net/biwidl13/yihshe/conda/etc/profile.d/conda.sh
conda activate pytcu10

# python3 -m tools.train_net --num-gpus 8 \
#         --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_base1.yaml

# python3 -m tools.ckpt_surgery \
#         --src1 checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1/model_final.pth \
#         --method randinit \
#         --save-dir checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_all1

python3 -m tools.train_net --num-gpus 1 \
        --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_1shot.yaml \
        --opts MODEL.WEIGHTS checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_all1/model_reset_surgery.pth