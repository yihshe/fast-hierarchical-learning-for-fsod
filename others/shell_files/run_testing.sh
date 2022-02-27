#!/bin/bash
#SBATCH  --output=sbatch_log/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=30G

source /scratch_net/biwidl13/yihshe/conda/etc/profile.d/conda.sh
conda activate pytcu10

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/PascalVOC-detection_CG/split1/faster_rcnn_R_101_FPN_ft_all1_1shot.yaml \
#         --eval-all
# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/COCO-detection_CG/faster_rcnn_R_101_FPN_ft_all_1shot.yaml \
#         --eval-all

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/LVIS-detection/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat.yaml \
#         --eval-only \
#         --opts MODEL.WEIGHTS checkpoints/lvis/faster_rcnn/faster_rcnn_R_101_FPN_base_norepeat_cos/model_reset_combine.pth \
#                OUTPUT_DIR checkpoints/lvis/faster_rcnn/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat

python3 -m tools.test_net --num-gpus 1 \
        --config-file configs/COCO-detection_HDA_CG_augmentation/faster_rcnn_R_101_FPN_ft_fc_all_1shot.yaml \
        --eval-only \
        --test-with-gt \
        --opts MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE 1000 \
               MODEL.WEIGHTS checkpoints_temp/coco_hda_cg_aug_infer/faster_rcnn/faster_rcnn_R_101_FPN_ft_fc_all_10shot/model_final.pth \
               OUTPUT_DIR checkpoints_temp/coco_hda_cg_aug_infer/faster_rcnn/faster_rcnn_R_101_FPN_ft_fc_all_10shot
