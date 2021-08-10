#!/bin/bash
#SBATCH  --output=sbatch_log/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=30G
#SBATCH  --constraint='titan_xp'

source /scratch_net/biwidl13/yihshe/conda/etc/profile.d/conda.sh
conda activate pytcu10

# VOC 
python3 -m tools.tune_net --num-gpus 1 \
        --config-file configs/PascalVOC-detection_CG/split1/faster_rcnn_R_101_FPN_ft_all1_1shot.yaml 

python3 -m tools.tune_net --num-gpus 1 \
        --config-file configs/PascalVOC-detection_CG/split1/faster_rcnn_R_101_FPN_ft_all1_5shot.yaml 

python3 -m tools.tune_net --num-gpus 1 \
        --config-file configs/PascalVOC-detection_CG/split1/faster_rcnn_R_101_FPN_ft_all1_10shot.yaml 

python3 -m tools.tune_net --num-gpus 1 \
        --config-file configs/PascalVOC-detection_CG/split2/faster_rcnn_R_101_FPN_ft_all2_1shot.yaml 

python3 -m tools.tune_net --num-gpus 1 \
        --config-file configs/PascalVOC-detection_CG/split2/faster_rcnn_R_101_FPN_ft_all2_5shot.yaml 

python3 -m tools.tune_net --num-gpus 1 \
        --config-file configs/PascalVOC-detection_CG/split2/faster_rcnn_R_101_FPN_ft_all2_10shot.yaml 

python3 -m tools.test_net --num-gpus 1 \
        --config-file configs/PascalVOC-detection_CG/split1/faster_rcnn_R_101_FPN_ft_all1_1shot.yaml \
        --eval-all

python3 -m tools.test_net --num-gpus 1 \
        --config-file configs/PascalVOC-detection_CG/split1/faster_rcnn_R_101_FPN_ft_all1_5shot.yaml \
        --eval-all

python3 -m tools.test_net --num-gpus 1 \
        --config-file configs/PascalVOC-detection_CG/split1/faster_rcnn_R_101_FPN_ft_all1_10shot.yaml \
        --eval-all

python3 -m tools.test_net --num-gpus 1 \
        --config-file configs/PascalVOC-detection_CG/split2/faster_rcnn_R_101_FPN_ft_all2_1shot.yaml \
        --eval-all

python3 -m tools.test_net --num-gpus 1 \
        --config-file configs/PascalVOC-detection_CG/split2/faster_rcnn_R_101_FPN_ft_all2_5shot.yaml \
        --eval-all

python3 -m tools.test_net --num-gpus 1 \
        --config-file configs/PascalVOC-detection_CG/split2/faster_rcnn_R_101_FPN_ft_all2_10shot.yaml \
        --eval-all
        
# COCO
# python3 -m tools.tune_net --num-gpus 1 \
#         --config-file configs/COCO-detection_CG/faster_rcnn_R_101_FPN_ft_all_1shot.yaml \
#         # --opts MODEL.WEIGHTS checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel_1shot_combine/model_reset_combine.pth

# # COCO novel weights
# python3 -m tools.tune_net --num-gpus 1 \
#         --config-file configs/COCO-detection_CG/faster_rcnn_R_101_FPN_ft_novel_1shot.yaml \
#         # --opts MODEL.WEIGHTS checkpoints_temp/CG_L2_RTS/coco/sgd2_shot1_novel_iter500_seed10/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel_1shot/model_0000099.pth
