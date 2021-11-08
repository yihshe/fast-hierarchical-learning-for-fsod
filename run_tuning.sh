#!/bin/bash
#SBATCH  --output=sbatch_log/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=30G

source /scratch_net/biwidl13/yihshe/conda/etc/profile.d/conda.sh
conda activate pytcu10

# VOC 
# python3 -m tools.tune_net --num-gpus 1 \
#         --config-file configs/PascalVOC-detection_CG/split1/faster_rcnn_R_101_FPN_ft_all1_1shot.yaml 

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/PascalVOC-detection_CG/split1/faster_rcnn_R_101_FPN_ft_all1_1shot.yaml \
#         --eval-all
        
# COCO
# python3 -m tools.tune_net_hda_new_setting --num-gpus 1 \
#         --config-file configs/COCO-detection_HDA_CG_new_setting/faster_rcnn_R_101_FPN_ft_fc_all_1shot.yaml 

# python3 -m tools.train_net_modified_hda_new_setting --num-gpus 1 \
#         --config-file configs/COCO-detection_TFA_SGD_new_setting/faster_rcnn_R_101_FPN_ft_all_1shot.yaml

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/COCO-detection_HDA_CG_new_setting/faster_rcnn_R_101_FPN_ft_fc_all_1shot.yaml \
#         --eval-all

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/COCO-detection_HDA_CG_new_setting/faster_rcnn_R_101_FPN_ft_fc_all_10shot.yaml \
#         --eval-only \
#         --resume \
#         --results-path checkpoints_HDA_CG_new_setting/coco/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_10shot_newfiltering_bbox_new_aug/inference/coco_instances_results.json

# LVIS weight initialization
# python3 -m tools.ckpt_surgery \
#         --src1 checkpoints/lvis/faster_rcnn/faster_rcnn_R_101_FPN_base_norepeat_cos/model_final.pth \
#         --method remove  --lvis

# python3 -m tools.ckpt_surgery \
#         --src1 checkpoints/lvis/faster_rcnn/faster_rcnn_R_101_FPN_base_norepeat_cos/model_final.pth \
#         --src2 checkpoints/lvis/faster_rcnn/faster_rcnn_R_101_FPN_fc_novel_cosine_norepeat/model_final.pth \
#         --method combine  --lvis 

# python3 -m tools.train_net --num-gpus 2 \
#         --config-file configs/LVIS-detection/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat.yaml \
#         --opts MODEL.WEIGHTS checkpoints/lvis/faster_rcnn/faster_rcnn_R_101_FPN_base_norepeat_cos/model_final.pth SOLVER.CHECKPOINT_PERIOD 10000 \
#         OUTPUT_DIR checkpoints/lvis/faster_rcnn/faster_rcnn_R_101_FPN_cosine_randinit_all_norepeat

# python3 -m tools.tune_net --num-gpus 1\
#         --config-file configs/LVIS-detection_SCG/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat.yaml \
#         --SCGTrainer \
#         --opts CG_PARAMS.NUM_NEWTON_ITER 1 \
#                CG_PARAMS.INIT_HESSIAN_REG 0.5 \
#                OUTPUT_DIR checkpoints_temp/lvis/faster_rcnn/faster_rcnn_R_101_FPN_cosine_randinit_all_norepeat_newton1_hessian

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/LVIS-detection_SCG/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat.yaml \
#         --eval-all \
#         --opts OUTPUT_DIR checkpoints_temp/lvis/faster_rcnn/faster_rcnn_R_101_FPN_cosine_randinit_all_norepeat_newton1_hessian

# python3 -m tools.tune_net --num-gpus 1\
#         --config-file configs/COCO-detection_SCG/faster_rcnn_R_101_FPN_ft_all_10shot.yaml \
#         --SCGTrainer \
#         --opts SOLVER.MAX_ITER 100 SOLVER.CHECKPOINT_PERIOD 10 SOLVER.IMS_PER_BATCH 10 SOLVER.NUM_BATCHES_PER_SET 30 \
#                CG_PARAMS.NUM_NEWTON_ITER 1 \
#                OUTPUT_DIR checkpoints_temp/COCO_TFA_SCG/coco/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_10shot_SCG_test_newton2

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/COCO-detection_SCG/faster_rcnn_R_101_FPN_ft_all_10shot.yaml \
#         --eval-all \
#         --opts OUTPUT_DIR checkpoints_temp/COCO_TFA_SCG/coco/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_10shot_SCG_test_newton2

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/LVIS-detection/faster_rcnn_R_101_FPN_base_norepeat_cosine.yaml \
#         --eval-only \
#         --opts MODEL.WEIGHTS checkpoints/lvis/faster_rcnn/faster_rcnn_R_101_FPN_base_norepeat_cos/model_final.pth \
#                OUTPUT_DIR checkpoints/lvis/faster_rcnn/faster_rcnn_R_101_FPN_base_norepeat_cos

# python3 -m tools.train_net_modified --num-gpus 1 \
#         --config-file configs/PascalVOC-detection_HDA_CG/split1/faster_rcnn_R_101_FPN_ft_all1_5shot.yaml \
#         --opts MODEL.WEIGHTS checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1/model_reset_surgery_ts.pth \
#                MODEL.ROI_HEADS.NAME TwoStageROIHeads MODEL.ROI_HEADS.OUTPUT_LAYER FastRCNNOutputLayers \
#                MODEL.PRETRAINED_BASE_MODEL checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1/model_final.pth \
#                SOLVER.MAX_ITER 20000 SOLVER.CHECKPOINT_PERIOD 20001 \
#                OUTPUT_DIR checkpoints_temp/voc_hda_sgd/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_5shot_randnovel

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/PascalVOC-detection_HDA_CG/split1/faster_rcnn_R_101_FPN_ft_all1_5shot.yaml \
#         --eval-all \
#         --opts MODEL.ROI_HEADS.NAME TwoStageROIHeads MODEL.ROI_HEADS.OUTPUT_LAYER FastRCNNOutputLayers \
#                OUTPUT_DIR checkpoints_temp/voc_hda_sgd/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_5shot_randnovel

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_5shot.yaml \
#         --eval-only \
#         --opts MODEL.WEIGHTS checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_5shot_randnovel/model_final.pth \
#                OUTPUT_DIR checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_5shot_randnovel

