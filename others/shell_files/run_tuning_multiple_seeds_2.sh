#!/bin/bash
#SBATCH  --output=sbatch_log/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=30G
#SBATCH  --constraint='titan_xp'

source /scratch_net/biwidl13/yihshe/conda/etc/profile.d/conda.sh
conda activate pytcu10

# COCO HDA SCG
# python3 -m tools.run_experiments_SCG --num-gpus 1 \
#         --shots 10 --seeds 0 10 \
#         --HDA --fc \
#         --coco --eval-saved-results

# COCO 
# TFA SGD feature extracted new setting
# first fine tune the novel weights
# python3 -m tools.run_experiments_SGD --num-gpus 1 \
#         --shots 1 2 3 5 10 --seeds 0 10 --lr 0.01 --ckpt-freq 1 \
#         --two-stage --novel-finetune --coco

# train the model
# python3 -m tools.run_experiments_SGD --num-gpus 1 \
#         --shots 1 2 3 5 10 --seeds 0 1 --lr 0.001 --ckpt-freq 1 \
#         --two-stage --coco

# python3 -m tools.run_experiments_SGD --num-gpus 1 \
#         --shots 10 --seeds 8 10 --lr 0.001 --ckpt-freq 1 \
#         --two-stage --coco \
#         --eval-saved-results

# python3 -m tools.tune_net_hda_new_setting --num-gpus 1 \
#         --config-file configs/COCO-detection_HDA_CG_new_setting/faster_rcnn_R_101_FPN_ft_fc_all_10shot.yaml \
#         --opts OUTPUT_DIR checkpoints_temp/coco_hda_cg_aug/rebuttal_10shot

python3 -m tools.tune_net --num-gpus 1 \
        --CGTrainer \
        --config-file configs/COCO-detection_HDA_CG_augmentation/faster_rcnn_R_101_FPN_ft_fc_all_10shot.yaml \
        --opts OUTPUT_DIR checkpoints_temp/coco_hda_cg_aug/rebuttal_10shot \

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/COCO-detection_HDA_CG_new_setting/faster_rcnn_R_101_FPN_ft_fc_all_10shot.yaml \
#         --eval-all

# python3 -m tools.tune_net --num-gpus 1\
#         --config-file configs/LVIS-detection_SCG/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat.yaml \
#         --SCGTrainer \
#         --opts MODEL.WEIGHTS checkpoints/lvis/faster_rcnn/faster_rcnn_R_101_FPN_base_norepeat_cos/model_reset_combine.pth \
#                SOLVER.MAX_ITER 200 SOLVER.CHECKPOINT_PERIOD 100 SOLVER.IMS_PER_BATCH 10 SOLVER.NUM_BATCHES_PER_SET 15 \
#                CG_PARAMS.NUM_NEWTON_ITER 2 \
#                CG_PARAMS.INIT_HESSIAN_REG 0.8 CG_PARAMS.HESSIAN_REG_FACTOR 0.9 \
#                OUTPUT_DIR checkpoints_temp/lvis/faster_rcnn/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat_newton2_hessian8e-1_9e-1_iter200

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/LVIS-detection_SCG/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat.yaml \
#         --eval-all \
#         --opts OUTPUT_DIR checkpoints_temp/lvis/faster_rcnn/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat_newton2_hessian8e-1_9e-1_iter200

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/LVIS-detection_HDA_SCG/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat.yaml \
#         --eval-only \
#         --test-with-gt \
#         --opts MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE 1000 \
#                MODEL.WEIGHTS checkpoints/lvis/faster_rcnn/faster_rcnn_R_101_FPN_base_norepeat_cos/model_reset_combine_hda_std_setting.pth \
#                OUTPUT_DIR checkpoints_temp/lvis_hda_scg/faster_rcnn/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat_infer_comb_init

# python3 -m tools.tune_net --num-gpus 1 \
#         --config-file configs/PascalVOC-detection_HDA_CG/split1/faster_rcnn_R_101_FPN_ft_fc_all1_10shot.yaml \
#         --CGTrainer \
#         --opts MODEL.WEIGHTS checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1/model_reset_surgery_ts.pth \
#                MODEL.ROI_HEADS.NAME TwoStageROIHeads MODEL.ROI_HEADS.OUTPUT_LAYER FastRCNNOutputLayers \
#                MODEL.PRETRAINED_BASE_MODEL checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1/model_final.pth \
#                SOLVER.CHECKPOINT_PERIOD 20 \
#                CG_PARAMS.NUM_NEWTON_ITER 60 CG_PARAMS.NUM_CG_ITER 2 CG_PARAMS.AUGMENTATION True \
#                CG_PARAMS.AUG_OPTIONS.PSEUDO_SHOTS 5 CG_PARAMS.AUG_OPTIONS.NOISE_LEVEL 0.1 CG_PARAMS.AUG_OPTIONS.DROP_RATE 0.5 \
#                OUTPUT_DIR checkpoints_temp/voc_hda_cg/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_10shot_randnovel_iter100_aug1e-1_5e-1

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/PascalVOC-detection_HDA_CG/split1/faster_rcnn_R_101_FPN_ft_fc_all1_10shot.yaml \
#         --eval-all \
#         --opts MODEL.ROI_HEADS.NAME TwoStageROIHeads MODEL.ROI_HEADS.OUTPUT_LAYER FastRCNNOutputLayers \
#                OUTPUT_DIR checkpoints_temp/voc_hda_cg/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_10shot_randnovel_iter100_aug1e-1_5e-1

# python3 -m tools.train_net_modified --num-gpus 1 \
#         --config-file configs/PascalVOC-detection_CG/split1/faster_rcnn_R_101_FPN_ft_all1_3shot.yaml \
#         --opts MODEL.WEIGHTS checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1/model_reset_surgery.pth \
#                MODEL.PRETRAINED_BASE_MODEL checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1/model_final.pth \
#                SOLVER.MAX_ITER 12000 SOLVER.CHECKPOINT_PERIOD 12001 \
#                OUTPUT_DIR checkpoints_temp/voc_tfa_sgd/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_3shot_randnovel

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/PascalVOC-detection_CG/split1/faster_rcnn_R_101_FPN_ft_all1_3shot.yaml \
#         --eval-only \
#         --opts MODEL.WEIGHTS checkpoints_temp/voc_tfa_sgd/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_3shot_randnovel/model_final.pth \
#                OUTPUT_DIR checkpoints_temp/voc_tfa_sgd/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_3shot_randnovel


# python3 -m tools.train_net_modified --num-gpus 1 \
#         --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_10shot.yaml \
#         --opts SOLVER.IMS_PER_BATCH_FEAT_EXTRACT 10 DATALOADER.ASPECT_RATIO_GROUPING False SOLVER.CHECKPOINT_PERIOD 40001 \
#                OUTPUT_DIR checkpoints_temp/voc_tfa_sgd_exp2/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_10shot_randnovel_5feats

# python3 -m tools.train_net_modified --num-gpus 1 \
#         --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_5shot.yaml \
#         --opts SOLVER.IMS_PER_BATCH_FEAT_EXTRACT 10 DATALOADER.ASPECT_RATIO_GROUPING False SOLVER.CHECKPOINT_PERIOD 20001 \
#                OUTPUT_DIR checkpoints_temp/voc_tfa_sgd_exp2/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_5shot_randnovel_5feats

# python3 -m tools.train_net_modified --num-gpus 1 \
#         --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_3shot.yaml \
#         --opts SOLVER.IMS_PER_BATCH_FEAT_EXTRACT 10 DATALOADER.ASPECT_RATIO_GROUPING False SOLVER.CHECKPOINT_PERIOD 12001 \
#                OUTPUT_DIR checkpoints_temp/voc_tfa_sgd_exp2/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_3shot_randnovel_5feats

# python3 -m tools.train_net_modified --num-gpus 1 \
#         --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_2shot.yaml \
#         --opts SOLVER.IMS_PER_BATCH_FEAT_EXTRACT 10 DATALOADER.ASPECT_RATIO_GROUPING False SOLVER.CHECKPOINT_PERIOD 8001 \
#                OUTPUT_DIR checkpoints_temp/voc_tfa_sgd_exp2/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_2shot_randnovel_5feats

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_5shot.yaml \
#         --eval-only \
#         --test-with-gt \
#         --opts MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE 1000 \
#                MODEL.WEIGHTS checkpoints_temp/voc_tfa_sgd_original/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_5shot_randnovel/model_final.pth \
#                OUTPUT_DIR checkpoints_temp/voc_infer/voc_tfa_sgd_original/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_5shot_randnovel

# python3 -m tools.train_net --num-gpus 1 \
#         --config-file configs/PascalVOC-detection_HDA_CG/split1/faster_rcnn_R_101_FPN_ft_fc_all1_5shot.yaml \
#         --opts MODEL.WEIGHTS checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1/model_reset_surgery_ts.pth \
#                MODEL.ROI_HEADS.NAME TwoStageROIHeads MODEL.ROI_HEADS.NUM_CLASSES_BASE 15 MODEL.ROI_HEADS.NUM_CLASSES_NOVEL 5 MODEL.ROI_HEADS.OUTPUT_LAYER FastRCNNOutputLayers MODEL.ROI_HEADS.SCORE_THRESH_TEST_SECOND 0.05 \
#                SOLVER.IMS_PER_BATCH 12 SOLVER.MAX_ITER 20000 SOLVER.CHECKPOINT_PERIOD 5000 SOLVER.WARMUP_ITERS 10 SOLVER.IMS_PER_BATCH_FEAT_EXTRACT 10 \
#                OUTPUT_DIR checkpoints_temp/voc_hda_sgd_batch/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_5shot_randnovel_l1_basethresh_5e-2

# python -m tools.test_net --num-gpus 1 \
#        --config-file configs/PascalVOC-detection_HDA_CG/split1/faster_rcnn_R_101_FPN_ft_fc_all1_5shot.yaml \
#        --eval-all \
#        --opts MODEL.ROI_HEADS.NAME TwoStageROIHeads MODEL.ROI_HEADS.NUM_CLASSES_BASE 15 MODEL.ROI_HEADS.NUM_CLASSES_NOVEL 5 MODEL.ROI_HEADS.OUTPUT_LAYER FastRCNNOutputLayers MODEL.ROI_HEADS.SCORE_THRESH_TEST_SECOND 0.05 \
#               OUTPUT_DIR checkpoints_temp/voc_hda_sgd_batch/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_5shot_randnovel_l1_basethresh_5e-2

# python3 -m tools.tune_net --num-gpus 1\
#         --config-file configs/COCO-detection_HDA_SCG/faster_rcnn_R_101_FPN_ft_fc_all_10shot.yaml \
#         --SCGTrainer \
#         --opts MODEL.WEIGHTS checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_all/model_reset_surgery_ts.pth \
#                SOLVER.MAX_ITER 300 SOLVER.CHECKPOINT_PERIOD 100 SOLVER.IMS_PER_BATCH 8 SOLVER.NUM_BATCHES_PER_SET 2 \
#                CG_PARAMS.NUM_NEWTON_ITER 1 CG_PARAMS.NUM_CG_ITER 2 \
#                CG_PARAMS.INIT_HESSIAN_REG 0.5 CG_PARAMS.HESSIAN_REG_FACTOR 1.0 \
#                OUTPUT_DIR checkpoints_temp/coco_hda_scg/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_10shot

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/COCO-detection_HDA_SCG/faster_rcnn_R_101_FPN_ft_fc_all_10shot.yaml \
#         --eval-all \
#         --opts OUTPUT_DIR checkpoints_temp/coco_hda_scg/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_10shot

# python3 -m tools.tune_net --num-gpus 1 \
#         --config-file configs/COCO-detection_HDA_CG_augmentation/faster_rcnn_R_101_FPN_ft_fc_all_3shot.yaml \
#         --CGTrainer \
#         --opts MODEL.WEIGHTS checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_all/model_reset_surgery_ts.pth \
#                SOLVER.CHECKPOINT_PERIOD 6 SOLVER.IMS_PER_BATCH_FEAT_EXTRACT 10 \
#                OUTPUT_DIR checkpoints_temp/coco_hda_cg_aug/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_3shot

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/COCO-detection_HDA_CG_augmentation/faster_rcnn_R_101_FPN_ft_fc_all_3shot.yaml \
#         --eval-all \
#         --opts OUTPUT_DIR checkpoints_temp/coco_hda_cg_aug/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_3shot

# python3 -m tools.train_net_modified --num-gpus 1 \
#         --config-file configs/COCO-detection_TFA_SGD/faster_rcnn_R_101_FPN_ft_all_3shot.yaml \
#         --opts MODEL.WEIGHTS checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_all/model_reset_surgery.pth \
#                SOLVER.MAX_ITER 300 SOLVER.CHECKPOINT_PERIOD 60 SOLVER.IMS_PER_BATCH_FEAT_EXTRACT 10 \
#                OUTPUT_DIR checkpoints_temp/coco_tfa_sgd/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_3shot_plus2

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/COCO-detection_TFA_SGD/faster_rcnn_R_101_FPN_ft_all_3shot.yaml \
#         --eval-all \
#         --opts OUTPUT_DIR checkpoints_temp/coco_tfa_sgd/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_3shot_plus2

# python3 -m tools.train_net --num-gpus 1 \
#         --config-file configs/COCO-detection_CG/faster_rcnn_R_101_FPN_ft_all_3shot.yaml \
#         --opts MODEL.WEIGHTS checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_all/model_reset_surgery.pth \
#                SOLVER.MAX_ITER 1000 SOLVER.CHECKPOINT_PERIOD 1001 \
#                OUTPUT_DIR checkpoints_temp/coco_tfa_sgd_time/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_3shot
