#!/bin/bash
#SBATCH  --output=sbatch_log/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=30G
#SBATCH  --constraint='titan_xp'

source /scratch_net/biwidl13/yihshe/conda/etc/profile.d/conda.sh
conda activate pytcu10

# COCO HDA SCG
# python3 -m tools.run_experiments_SCG --num-gpus 1 \
#         --shots 3 5 --seeds 0 10 \
#         --HDA --fc \
#         --coco --eval-saved-results

# COCO 
# HDA new setting 
# python3 -m tools.run_experiments_CG --num-gpus 1 \
#         --shots 5 10 --seeds 0 10 --ckpt-freq 1 \
#         --HDA --fc \
#         --coco \
#         --augmentation \
#         --eval-saved-results

# TFA weight regvec
# python3 -m tools.run_experiments_CG --num-gpus 1 \
#         --shots 1 2 3 5 10 --seeds 0 10 --ckpt-freq 1 \
#         --coco

# python3 -m tools.tune_net_hda_new_setting --num-gpus 1 \
#         --config-file configs/COCO-detection_HDA_CG_new_setting/faster_rcnn_R_101_FPN_ft_fc_all_5shot.yaml 

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/COCO-detection_HDA_CG_new_setting/faster_rcnn_R_101_FPN_ft_fc_all_5shot.yaml \
#         --eval-all

# python3 -m tools.train_net_modified_hda_new_setting --num-gpus 1 \
#         --config-file configs/COCO-detection_HDA_SGD_new_setting/faster_rcnn_R_101_FPN_ft_all_5shot.yaml

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/COCO-detection_HDA_SGD_new_setting/faster_rcnn_R_101_FPN_ft_all_5shot.yaml \
#         --eval-all

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/COCO-detection_TFA_SGD_new_setting/faster_rcnn_R_101_FPN_ft_all_5shot.yaml \
#         --eval-only \
#         --eval-iter 80000 \
#         --test-with-gt

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/COCO-detection_TFA_SGD_new_setting/faster_rcnn_R_101_FPN_ft_all_1shot.yaml \
#         --eval-only \
#         --eval-iter 16000 \
#         --test-with-gt

# python3 -m tools.tune_net --num-gpus 1\
#         --config-file configs/LVIS-detection_SCG/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat.yaml \
#         --SCGTrainer \
#         --opts MODEL.WEIGHTS checkpoints/lvis/faster_rcnn/faster_rcnn_R_101_FPN_base_norepeat_cos/model_reset_combine.pth \
#                CG_PARAMS.NUM_NEWTON_ITER 2 \
#                CG_PARAMS.INIT_HESSIAN_REG 1.0 CG_PARAMS.HESSIAN_REG_FACTOR 0.9 \
#                OUTPUT_DIR checkpoints_temp/lvis/faster_rcnn/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat_newton2_hessian1_9e-1

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/LVIS-detection_SCG/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat.yaml \
#         --eval-all \
#         --opts OUTPUT_DIR checkpoints_temp/lvis/faster_rcnn/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat_newton2_hessian1_9e-1

# python3 -m tools.tune_net --num-gpus 1\
#         --config-file configs/LVIS-detection_HDA_SCG/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat.yaml \
#         --SCGTrainer \
#         --opts MODEL.WEIGHTS checkpoints/lvis/faster_rcnn/faster_rcnn_R_101_FPN_base_norepeat_cos/model_reset_combine_hda_std_setting_rc.pth \
#                MODEL.ROI_HEADS.NAME TwoStageROIHeads_lvis_rc MODEL.ROI_HEADS.NUM_CLASSES_BASE 315 MODEL.ROI_HEADS.NUM_CLASSES_NOVEL 915 \
#                SOLVER.MAX_ITER 500 SOLVER.CHECKPOINT_PERIOD 100 SOLVER.IMS_PER_BATCH 8 SOLVER.NUM_BATCHES_PER_SET 2 \
#                CG_PARAMS.NUM_NEWTON_ITER 1 \
#                CG_PARAMS.INIT_HESSIAN_REG 0.5 CG_PARAMS.HESSIAN_REG_FACTOR 1.0 \
#                OUTPUT_DIR checkpoints_temp/lvis_hda_scg/faster_rcnn/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat_newton1_hessian5e-1_iter500_rc

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/LVIS-detection_HDA_SCG/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat.yaml \
#         --eval-all \
#         --opts MODEL.ROI_HEADS.NAME TwoStageROIHeads_lvis_rc MODEL.ROI_HEADS.NUM_CLASSES_BASE 315 MODEL.ROI_HEADS.NUM_CLASSES_NOVEL 915 \
#                OUTPUT_DIR checkpoints_temp/lvis_hda_scg/faster_rcnn/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat_newton1_hessian5e-1_iter500_rc

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/LVIS-detection/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat.yaml \
#         --eval-only \
#         --test-with-gt \
#         --opts MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE 1000 \
#                MODEL.ROI_HEADS.NAME TwoStageROIHeads_lvis_rc MODEL.ROI_HEADS.NUM_CLASSES_BASE 315 MODEL.ROI_HEADS.NUM_CLASSES_NOVEL 915 \
#                MODEL.WEIGHTS checkpoints_temp/lvis_hda_scg/faster_rcnn/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat_newton1_hessian5e-1_iter500_rc/model_final.pth \
#                OUTPUT_DIR checkpoints_temp/lvis_hda_scg/faster_rcnn/faster_rcnn_R_101_FPN_cosine_combined_all_infer_final_exp2_rc

# # Pascal VOC
# python3 -m tools.tune_net --num-gpus 1 \
#         --config-file configs/PascalVOC-detection_HDA_CG/split1/faster_rcnn_R_101_FPN_ft_fc_all1_3shot.yaml \
#         --CGTrainer \
#         --opts MODEL.WEIGHTS checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1/model_reset_surgery_ts.pth \
#                MODEL.ROI_HEADS.NAME TwoStageROIHeads MODEL.ROI_HEADS.OUTPUT_LAYER FastRCNNOutputLayers \
#                MODEL.PRETRAINED_BASE_MODEL checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1/model_final.pth \
#                SOLVER.CHECKPOINT_PERIOD 20 \
#                CG_PARAMS.NUM_NEWTON_ITER 60 CG_PARAMS.NUM_CG_ITER 2 CG_PARAMS.AUGMENTATION False \
#                OUTPUT_DIR checkpoints_temp/voc_hda_cg/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_3shot_randnovel_iter100

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/PascalVOC-detection_HDA_CG/split1/faster_rcnn_R_101_FPN_ft_fc_all1_3shot.yaml \
#         --eval-all \
#         --opts MODEL.ROI_HEADS.NAME TwoStageROIHeads MODEL.ROI_HEADS.OUTPUT_LAYER FastRCNNOutputLayers \
#                OUTPUT_DIR checkpoints_temp/voc_hda_cg/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_3shot_randnovel_iter100

# python3 -m tools.train_net_modified --num-gpus 1 \
#         --config-file configs/PascalVOC-detection_CG/split1/faster_rcnn_R_101_FPN_ft_all1_2shot.yaml \
#         --opts MODEL.WEIGHTS checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1/model_reset_surgery.pth \
#                MODEL.PRETRAINED_BASE_MODEL checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1/model_final.pth \
#                SOLVER.MAX_ITER 8000 SOLVER.CHECKPOINT_PERIOD 8001 \
#                OUTPUT_DIR checkpoints_temp/voc_tfa_sgd/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_2shot_randnovel

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/PascalVOC-detection_CG/split1/faster_rcnn_R_101_FPN_ft_all1_2shot.yaml \
#         --eval-only \
#         --opts MODEL.WEIGHTS checkpoints_temp/voc_tfa_sgd/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_2shot_randnovel/model_final.pth \
#                OUTPUT_DIR checkpoints_temp/voc_tfa_sgd/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_2shot_randnovel

# python3 -m tools.train_net_modified --num-gpus 1 \
#         --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_2shot.yaml \
#         --opts SOLVER.IMS_PER_BATCH_FEAT_EXTRACT 10 DATALOADER.ASPECT_RATIO_GROUPING False SOLVER.CHECKPOINT_PERIOD 8001 \
#                OUTPUT_DIR checkpoints_temp/voc_tfa_sgd_exp3/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_2shot_randnovel

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/PascalVOC-detection_HDA_CG/split1/faster_rcnn_R_101_FPN_ft_fc_all1_5shot.yaml \
#         --eval-only \
#         --test-with-gt \
#         --opts MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE 1000 \
#                MODEL.ROI_HEADS.NAME TwoStageROIHeads MODEL.ROI_HEADS.OUTPUT_LAYER FastRCNNOutputLayers MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.01 MODEL.ROI_HEADS.SCORE_THRESH_TEST_SECOND 0.05 \
#                MODEL.WEIGHTS checkpoints_temp/voc_hda_cg/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_5shot_randnovel_iter100/model_final.pth \
#                OUTPUT_DIR checkpoints_temp/voc_infer/voc_hda_cg/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_5shot_randnovel_sameModel_novelthresh_1e-2

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/PascalVOC-detection_HDA_CG/split1/faster_rcnn_R_101_FPN_ft_fc_all1_5shot.yaml \
#         --eval-only \
#         --test-with-gt \
#         --opts MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE 1000 \
#                MODEL.ROI_HEADS.NAME TwoStageROIHeads MODEL.ROI_HEADS.OUTPUT_LAYER FastRCNNOutputLayers MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.005 MODEL.ROI_HEADS.SCORE_THRESH_TEST_SECOND 0.05 \
#                MODEL.WEIGHTS checkpoints_temp/voc_hda_cg/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_5shot_randnovel_iter100/model_final.pth \
#                OUTPUT_DIR checkpoints_temp/voc_infer/voc_hda_cg/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_5shot_randnovel_sameModel_novelthresh_5e-3

# python3 -m tools.tune_net --num-gpus 1 \
#         --config-file configs/PascalVOC-detection_HDA_CG/split1/faster_rcnn_R_101_FPN_ft_fc_all1_5shot.yaml \
#         --SCGTrainer \
#         --opts MODEL.WEIGHTS checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1/model_reset_surgery_ts.pth \
#                MODEL.ROI_HEADS.NAME TwoStageROIHeads MODEL.ROI_HEADS.NUM_CLASSES_BASE 15 MODEL.ROI_HEADS.NUM_CLASSES_NOVEL 5 MODEL.ROI_HEADS.OUTPUT_LAYER FastRCNNOutputLayers MODEL.ROI_HEADS.SCORE_THRESH_TEST_SECOND 0.5 \
#                DATALOADER.ASPECT_RATIO_GROUPING True \
#                SOLVER.IMS_PER_BATCH 8 SOLVER.NUM_BATCHES_PER_SET 2 SOLVER.MAX_ITER 2000 SOLVER.CHECKPOINT_PERIOD 1000 \
#                CG_PARAMS.NUM_NEWTON_ITER 1 CG_PARAMS.NUM_CG_ITER 2 CG_PARAMS.INIT_HESSIAN_REG 0.5 CG_PARAMS.HESSIAN_REG_FACTOR 1.0 \
#                OUTPUT_DIR checkpoints_temp/voc_hda_scg/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_5shot_randnovel_l1_cgiter2_basethresh_5e-1

# COCO HDA SCG
# python3 -m tools.tune_net --num-gpus 1\
#         --config-file configs/COCO-detection_HDA_SCG/faster_rcnn_R_101_FPN_ft_fc_all_5shot.yaml \
#         --SCGTrainer \
#         --opts MODEL.WEIGHTS checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_all/model_reset_surgery_ts.pth \
#                SOLVER.MAX_ITER 300 SOLVER.CHECKPOINT_PERIOD 100 SOLVER.IMS_PER_BATCH 8 SOLVER.NUM_BATCHES_PER_SET 2 \
#                CG_PARAMS.NUM_NEWTON_ITER 1 CG_PARAMS.NUM_CG_ITER 2 \
#                CG_PARAMS.INIT_HESSIAN_REG 0.5 CG_PARAMS.HESSIAN_REG_FACTOR 1.0 \
#                OUTPUT_DIR checkpoints_temp/coco_hda_scg/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_5shot

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/COCO-detection_HDA_SCG/faster_rcnn_R_101_FPN_ft_fc_all_5shot.yaml \
#         --eval-all \
#         --opts OUTPUT_DIR checkpoints_temp/coco_hda_scg/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_5shot

# python3 -m tools.tune_net --num-gpus 1 \
#         --config-file configs/COCO-detection_HDA_CG_augmentation/faster_rcnn_R_101_FPN_ft_fc_all_2shot.yaml \
#         --CGTrainer \
#         --opts MODEL.WEIGHTS checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_all/model_reset_surgery_ts.pth \
#                SOLVER.CHECKPOINT_PERIOD 6 SOLVER.IMS_PER_BATCH_FEAT_EXTRACT 10 \
#                OUTPUT_DIR checkpoints_temp/coco_hda_cg_aug/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_2shot

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/COCO-detection_HDA_CG_augmentation/faster_rcnn_R_101_FPN_ft_fc_all_2shot.yaml \
#         --eval-all \
#         --opts OUTPUT_DIR checkpoints_temp/coco_hda_cg_aug/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_2shot

# python3 -m tools.train_net_modified --num-gpus 1 \
#         --config-file configs/COCO-detection_TFA_SGD/faster_rcnn_R_101_FPN_ft_all_2shot.yaml \
#         --opts MODEL.WEIGHTS checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_all/model_reset_surgery.pth \
#                SOLVER.MAX_ITER 200 SOLVER.CHECKPOINT_PERIOD 40 SOLVER.IMS_PER_BATCH_FEAT_EXTRACT 10 \
#                OUTPUT_DIR checkpoints_temp/coco_tfa_sgd/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_2shot_plus2

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/COCO-detection_TFA_SGD/faster_rcnn_R_101_FPN_ft_all_2shot.yaml \
#         --eval-all \
#         --opts OUTPUT_DIR checkpoints_temp/coco_tfa_sgd/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_2shot_plus2

python3 -m tools.train_net --num-gpus 1 \
        --config-file configs/COCO-detection_CG/faster_rcnn_R_101_FPN_ft_all_2shot.yaml \
        --opts MODEL.WEIGHTS checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_all/model_reset_surgery.pth \
               SOLVER.MAX_ITER 1000 SOLVER.CHECKPOINT_PERIOD 1001 \
               OUTPUT_DIR checkpoints_temp/coco_tfa_sgd_time/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_2shot
