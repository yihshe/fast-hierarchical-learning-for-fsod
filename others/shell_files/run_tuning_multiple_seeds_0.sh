#!/bin/bash
#SBATCH  --output=sbatch_log/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=30G
#SBATCH  --constraint='titan_xp'

source /scratch_net/biwidl13/yihshe/conda/etc/profile.d/conda.sh
conda activate pytcu10

# COCO HDA SCG
# python3 -m tools.run_experiments_SCG --num-gpus 1 \
#         --shots 1 2 --seeds 0 10 \
#         --HDA --fc \
#         --coco --eval-saved-results

# python3 -m tools.run_experiments_CG --num-gpus 1 \
#         --shots 3 5 10 --seeds 1 10 --ckpt-freq 1 \
#         --HDA --fc \
#         --coco

# TFA weight regvec
# python3 -m tools.run_experiments_CG --num-gpus 1 \
#         --shots 1 2 3 5 10 --seeds 0 10 --ckpt-freq 1 \
#         --coco

# python3 -m tools.tune_net_hda_new_setting --num-gpus 1 \
#         --config-file configs/COCO-detection_HDA_CG_new_setting/faster_rcnn_R_101_FPN_ft_fc_all_3shot.yaml 

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/COCO-detection_HDA_CG_new_setting/faster_rcnn_R_101_FPN_ft_fc_all_3shot.yaml \
#         --eval-all

# python3 -m tools.train_net_modified_hda_new_setting --num-gpus 1 \
#         --config-file configs/COCO-detection_HDA_SGD_new_setting/faster_rcnn_R_101_FPN_ft_all_1shot.yaml

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/COCO-detection_HDA_SGD_new_setting/faster_rcnn_R_101_FPN_ft_all_1shot.yaml \
#         --eval-all

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/COCO-detection_HDA_CG_new_setting/faster_rcnn_R_101_FPN_ft_fc_all_1shot.yaml \
#         --eval-only \
#         --eval-iter 40 \
#         --test-with-gt

# # LVIS
# python3 -m tools.tune_net --num-gpus 1\
#         --config-file configs/LVIS-detection_SCG/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat.yaml \
#         --SCGTrainer \
#         --opts MODEL.WEIGHTS checkpoints/lvis/faster_rcnn/faster_rcnn_R_101_FPN_base_norepeat_cos/model_reset_combine.pth \
#                SOLVER.MAX_ITER 1000 SOLVER.CHECKPOINT_PERIOD 100 SOLVER.IMS_PER_BATCH 10 SOLVER.NUM_BATCHES_PER_SET 2 \
#                CG_PARAMS.NUM_NEWTON_ITER 1 \
#                CG_PARAMS.INIT_HESSIAN_REG 0.5 CG_PARAMS.HESSIAN_REG_FACTOR 1.0 \
#                OUTPUT_DIR checkpoints_temp/lvis/faster_rcnn/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat_newton1_hessian5e-1_iter1000

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/LVIS-detection_SCG/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat.yaml \
#         --eval-all \
#         --opts OUTPUT_DIR checkpoints_temp/lvis/faster_rcnn/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat_newton1_hessian5e-1_iter1000


# python3 -m tools.tune_net --num-gpus 1 \
#         --config-file configs/LVIS-detection_HDA_SCG/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat.yaml \
#         --SCGTrainer \
#         --opts MODEL.WEIGHTS checkpoints/lvis/faster_rcnn/faster_rcnn_R_101_FPN_base_norepeat_cos/model_reset_combine_hda_std_setting.pth \
#                MODEL.ROI_HEADS.NUM_CLASSES 1230 \
#                SOLVER.MAX_ITER 500 SOLVER.CHECKPOINT_PERIOD 100 SOLVER.IMS_PER_BATCH 8 SOLVER.NUM_BATCHES_PER_SET 2 \
#                CG_PARAMS.NUM_NEWTON_ITER 1 \
#                CG_PARAMS.INIT_HESSIAN_REG 0.5 CG_PARAMS.HESSIAN_REG_FACTOR 1.0 \
#                OUTPUT_DIR checkpoints_temp/lvis_hda_scg/faster_rcnn/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat_newton1_hessian5e-1_iter1000_combinit

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/LVIS-detection_HDA_SCG/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat.yaml \
#         --eval-all \
#         --opts OUTPUT_DIR checkpoints_temp/lvis_hda_scg/faster_rcnn/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat_newton1_hessian5e-1_iter1000_combinit

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/LVIS-detection_HDA_SCG/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat.yaml \
#         --eval-only \
#         --test-with-gt \
#         --opts MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE 1000 \
#                MODEL.WEIGHTS checkpoints_temp/lvis_hda_scg/faster_rcnn/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat_newton1_hessian5e-1_iter500_combinit/model_final.pth \
#                OUTPUT_DIR checkpoints_temp/lvis_hda_scg/faster_rcnn/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat_infer_final_exp1_combinit

# # Pascal VOC
# python3 -m tools.tune_net --num-gpus 1 \
#         --config-file configs/PascalVOC-detection_HDA_CG/split1/faster_rcnn_R_101_FPN_ft_fc_all1_2shot.yaml \
#         --CGTrainer \
#         --opts MODEL.WEIGHTS checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1/model_reset_surgery_ts.pth \
#                MODEL.ROI_HEADS.NAME TwoStageROIHeads MODEL.ROI_HEADS.OUTPUT_LAYER FastRCNNOutputLayers \
#                MODEL.PRETRAINED_BASE_MODEL checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1/model_final.pth \
#                SOLVER.CHECKPOINT_PERIOD 20 \
#                CG_PARAMS.NUM_NEWTON_ITER 60 CG_PARAMS.NUM_CG_ITER 2 CG_PARAMS.AUGMENTATION False \
#                OUTPUT_DIR checkpoints_temp/voc_hda_cg/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_2shot_randnovel_iter100_exp1

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/PascalVOC-detection_HDA_CG/split1/faster_rcnn_R_101_FPN_ft_fc_all1_2shot.yaml \
#         --eval-all \
#         --opts MODEL.ROI_HEADS.NAME TwoStageROIHeads MODEL.ROI_HEADS.OUTPUT_LAYER FastRCNNOutputLayers \
#                OUTPUT_DIR checkpoints_temp/voc_hda_cg/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_2shot_randnovel_iter100_exp1

# COCO HDA SCG
# python3 -m tools.tune_net --num-gpus 1\
#         --config-file configs/COCO-detection_HDA_SCG/faster_rcnn_R_101_FPN_ft_fc_all_1shot.yaml \
#         --SCGTrainer \
#         --opts MODEL.WEIGHTS checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_all/model_reset_surgery_ts.pth \
#                SOLVER.MAX_ITER 300 SOLVER.CHECKPOINT_PERIOD 100 SOLVER.IMS_PER_BATCH 8 SOLVER.NUM_BATCHES_PER_SET 2 \
#                CG_PARAMS.NUM_NEWTON_ITER 1 CG_PARAMS.NUM_CG_ITER 2 \
#                CG_PARAMS.INIT_HESSIAN_REG 0.5 CG_PARAMS.HESSIAN_REG_FACTOR 1.0 \
#                OUTPUT_DIR checkpoints_temp/coco_hda_scg/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/COCO-detection_HDA_SCG/faster_rcnn_R_101_FPN_ft_fc_all_1shot.yaml \
#         --eval-all \
#         --opts OUTPUT_DIR checkpoints_temp/coco_hda_scg/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot

# python3 -m tools.tune_net --num-gpus 1 \
#         --config-file configs/COCO-detection_CG/faster_rcnn_R_101_FPN_ft_all_1shot.yaml \
#         --CGTrainer \
#         --opts OUTPUT_DIR checkpoints_temp/coco_hda_cg/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot_SCG_test

# python3 -m tools.tune_net --num-gpus 1 \
#         --config-file configs/COCO-detection_HDA_CG_augmentation/faster_rcnn_R_101_FPN_ft_fc_all_1shot.yaml \
#         --CGTrainer \
#         --opts MODEL.WEIGHTS checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_all/model_reset_surgery_ts.pth \
#                SOLVER.CHECKPOINT_PERIOD 6 SOLVER.IMS_PER_BATCH_FEAT_EXTRACT 10 \
#                OUTPUT_DIR checkpoints_temp/coco_hda_cg_aug/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/COCO-detection_HDA_CG_augmentation/faster_rcnn_R_101_FPN_ft_fc_all_1shot.yaml \
#         --eval-all \
#         --opts OUTPUT_DIR checkpoints_temp/coco_hda_cg_aug/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot

# python3 -m tools.train_net_modified --num-gpus 1 \
#         --config-file configs/COCO-detection_TFA_SGD/faster_rcnn_R_101_FPN_ft_all_1shot.yaml \
#         --opts MODEL.WEIGHTS checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_all/model_reset_surgery.pth \
#                SOLVER.MAX_ITER 100 SOLVER.CHECKPOINT_PERIOD 20 SOLVER.IMS_PER_BATCH_FEAT_EXTRACT 10 \
#                OUTPUT_DIR checkpoints_temp/coco_tfa_sgd/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot_plus2

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/COCO-detection_TFA_SGD/faster_rcnn_R_101_FPN_ft_all_1shot.yaml \
#         --eval-all \
#         --opts OUTPUT_DIR checkpoints_temp/coco_tfa_sgd/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot_plus2

python3 -m tools.train_net --num-gpus 1 \
        --config-file configs/COCO-detection_CG/faster_rcnn_R_101_FPN_ft_all_1shot.yaml \
        --opts MODEL.WEIGHTS checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_all/model_reset_surgery.pth \
               SOLVER.MAX_ITER 1000 SOLVER.CHECKPOINT_PERIOD 1001 \
               OUTPUT_DIR checkpoints_temp/coco_tfa_sgd_time/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot
