#!/bin/bash
#SBATCH  --output=sbatch_log/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=30G
#SBATCH  --constraint='titan_xp'

source /scratch_net/biwidl13/yihshe/conda/etc/profile.d/conda.sh
conda activate pytcu10

# python3 -m tools.run_experiments_SGD --num-gpus 1 \
#         --shots 10 --seeds 0 10 --lr 0.001 --ckpt-freq 1 \
#         --two-stage --coco

# python3 -m tools.train_net_modified --num-gpus 1 \
#         --config-file configs/LVIS-detection/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat.yaml \
#         --opts MODEL.WEIGHTS checkpoints/lvis/faster_rcnn/faster_rcnn_R_101_FPN_base_norepeat_cos/model_reset_combine.pth SOLVER.CHECKPOINT_PERIOD 10000 \
#         OUTPUT_DIR checkpoints/lvis/faster_rcnn/faster_rcnn_R_101_FPN_cosine_combined_all_norepeat_test

# python3 -m tools.train_net_modified --num-gpus 1 \
#         --config-file configs/PascalVOC-detection_CG/split1/faster_rcnn_R_101_FPN_ft_all1_10shot.yaml \
#         --opts MODEL.WEIGHTS checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1/model_reset_surgery.pth \
#                MODEL.PRETRAINED_BASE_MODEL checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1/model_final.pth \
#                SOLVER.MAX_ITER 40000 SOLVER.CHECKPOINT_PERIOD 40001 \
#                OUTPUT_DIR checkpoints_temp/voc_tfa_sgd/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_10shot_randnovel

# python3 -m tools.tune_net --num-gpus 1 \
#         --config-file configs/PascalVOC-detection_HDA_CG/split1/faster_rcnn_R_101_FPN_ft_fc_all1_5shot.yaml \
#         --SCGTrainer \
#         --opts MODEL.WEIGHTS checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1/model_reset_surgery_ts.pth \
#                MODEL.ROI_HEADS.NAME TwoStageROIHeads MODEL.ROI_HEADS.OUTPUT_LAYER FastRCNNOutputLayers \
#                MODEL.PRETRAINED_BASE_MODEL checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1/model_final.pth \
#                SOLVER.MAX_ITER 500 SOLVER.CHECKPOINT_PERIOD 501 SOLVER.IMS_PER_BATCH 8 SOLVER.NUM_BATCHES_PER_SET 2 \
#                CG_PARAMS.NUM_NEWTON_ITER 1 CG_PARAMS.NUM_CG_ITER 2 CG_PARAMS.AUGMENTATION True \
#                CG_PARAMS.INIT_HESSIAN_REG 0.0 CG_PARAMS.HESSIAN_REG_FACTOR 1.0 \
#                OUTPUT_DIR checkpoints_temp/voc_hda_scg/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_5shot_randnovel_aug

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/PascalVOC-detection_HDA_CG/split1/faster_rcnn_R_101_FPN_ft_fc_all1_5shot.yaml \
#         --eval-all \
#         --opts MODEL.ROI_HEADS.NAME TwoStageROIHeads MODEL.ROI_HEADS.OUTPUT_LAYER FastRCNNOutputLayers \
#                OUTPUT_DIR checkpoints_temp/voc_hda_scg/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_5shot_randnovel_aug

# python3 -m tools.train_net_modified --num-gpus 1 \
#         --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_10shot.yaml \
#         --opts SOLVER.IMS_PER_BATCH_FEAT_EXTRACT 10 DATALOADER.ASPECT_RATIO_GROUPING False SOLVER.CHECKPOINT_PERIOD 40001 \
#                OUTPUT_DIR checkpoints_temp/voc_tfa_sgd_exp3/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_10shot_randnovel

# python3 -m tools.train_net --num-gpus 1 \
#         --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_fc_all1_3shot.yaml \
#         --opts MODEL.WEIGHTS checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1/model_reset_surgery_ts.pth \
#                MODEL.ROI_HEADS.NAME TwoStageROIHeads MODEL.ROI_HEADS.NUM_CLASSES_BASE 15 MODEL.ROI_HEADS.NUM_CLASSES_NOVEL 5 MODEL.ROI_HEADS.OUTPUT_LAYER FastRCNNOutputLayers MODEL.ROI_HEADS.SCORE_THRESH_TEST_SECOND 0.5 \
#                SOLVER.IMS_PER_BATCH 16 SOLVER.MAX_ITER 12000 SOLVER.CHECKPOINT_PERIOD 12001 SOLVER.WARMUP_ITERS 10 \
#                OUTPUT_DIR checkpoints_temp/voc_hda_sgd_batch/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_3shot_randnovel_l1_basethresh_5e-1_batch16

# python -m tools.test_net --num-gpus 1 \
#        --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_fc_all1_3shot.yaml \
#        --eval-all \
#        --opts MODEL.ROI_HEADS.NAME TwoStageROIHeads MODEL.ROI_HEADS.NUM_CLASSES_BASE 15 MODEL.ROI_HEADS.NUM_CLASSES_NOVEL 5 MODEL.ROI_HEADS.OUTPUT_LAYER FastRCNNOutputLayers MODEL.ROI_HEADS.SCORE_THRESH_TEST_SECOND 0.5 \
#               OUTPUT_DIR checkpoints_temp/voc_hda_sgd_batch/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_3shot_randnovel_l1_basethresh_5e-1_batch16

# python3 -m tools.tune_net --num-gpus 1\
#         --config-file configs/COCO-detection_HDA_SCG/faster_rcnn_R_101_FPN_ft_fc_all_5shot.yaml \
#         --SCGTrainer \
#         --opts MODEL.WEIGHTS checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_all/model_reset_surgery_ts.pth \
#                SOLVER.MAX_ITER 300 SOLVER.CHECKPOINT_PERIOD 100 SOLVER.IMS_PER_BATCH 8 SOLVER.NUM_BATCHES_PER_SET 2 \
#                CG_PARAMS.NUM_NEWTON_ITER 1 CG_PARAMS.NUM_CG_ITER 2 \
#                CG_PARAMS.INIT_HESSIAN_REG 0.5 CG_PARAMS.HESSIAN_REG_FACTOR 1.0 CG_PARAMS.AUGMENTATION True \
#                OUTPUT_DIR checkpoints_temp/coco_hda_scg/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_5shot_aug

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/COCO-detection_HDA_SCG/faster_rcnn_R_101_FPN_ft_fc_all_5shot.yaml \
#         --eval-all \
#         --opts OUTPUT_DIR checkpoints_temp/coco_hda_scg/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_5shot_aug

# python3 -m tools.train_net_modified --num-gpus 1 \
#         --config-file configs/COCO-detection_TFA_SGD/faster_rcnn_R_101_FPN_ft_all_1shot.yaml \
#         --opts MODEL.WEIGHTS checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_all/model_reset_surgery.pth \
#                SOLVER.MAX_ITER 16000 SOLVER.CHECKPOINT_PERIOD 2000 SOLVER.IMS_PER_BATCH_FEAT_EXTRACT 10 \
#                OUTPUT_DIR checkpoints_temp/coco_tfa_sgd/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot


# python3 -m tools.train_net_modified --num-gpus 1 \
#         --config-file configs/COCO-detection_TFA_SGD/faster_rcnn_R_101_FPN_ft_all_2shot.yaml \
#         --opts MODEL.WEIGHTS checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_all/model_reset_surgery.pth \
#                SOLVER.MAX_ITER 32000 SOLVER.CHECKPOINT_PERIOD 4000 SOLVER.IMS_PER_BATCH_FEAT_EXTRACT 10 \
#                OUTPUT_DIR checkpoints_temp/coco_tfa_sgd/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_2shot

# python3 -m tools.train_net_modified --num-gpus 1 \
#         --config-file configs/COCO-detection_TFA_SGD/faster_rcnn_R_101_FPN_ft_all_3shot.yaml \
#         --opts MODEL.WEIGHTS checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_all/model_reset_surgery.pth \
#                SOLVER.MAX_ITER 48000 SOLVER.CHECKPOINT_PERIOD 6000 SOLVER.IMS_PER_BATCH_FEAT_EXTRACT 10 \
#                OUTPUT_DIR checkpoints_temp/coco_tfa_sgd/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_3shot

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/COCO-detection_TFA_SGD/faster_rcnn_R_101_FPN_ft_all_1shot.yaml \
#         --eval-all \
#         --opts OUTPUT_DIR checkpoints_temp/coco_tfa_sgd/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_1shot

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/COCO-detection_TFA_SGD/faster_rcnn_R_101_FPN_ft_all_2shot.yaml \
#         --eval-all \
#         --opts OUTPUT_DIR checkpoints_temp/coco_tfa_sgd/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_2shot

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/COCO-detection_TFA_SGD/faster_rcnn_R_101_FPN_ft_all_3shot.yaml \
#         --eval-all \
#         --opts OUTPUT_DIR checkpoints_temp/coco_tfa_sgd/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_3shot

# python3 -m tools.tune_net --num-gpus 1 \
#         --config-file configs/COCO-detection_HDA_CG_augmentation/faster_rcnn_R_101_FPN_ft_fc_all_10shot.yaml \
#         --CGTrainer \
#         --opts MODEL.WEIGHTS checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_all/model_reset_surgery_ts.pth \
#                SOLVER.CHECKPOINT_PERIOD 6 SOLVER.IMS_PER_BATCH_FEAT_EXTRACT 10 \
#                OUTPUT_DIR checkpoints_temp/coco_hda_cg_aug/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_10shot

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/COCO-detection_HDA_CG_augmentation/faster_rcnn_R_101_FPN_ft_fc_all_10shot.yaml \
#         --eval-all \
#         --opts OUTPUT_DIR checkpoints_temp/coco_hda_cg_aug/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_10shot

# python3 -m tools.train_net_modified --num-gpus 1 \
#         --config-file configs/COCO-detection_TFA_SGD/faster_rcnn_R_101_FPN_ft_all_10shot.yaml \
#         --opts MODEL.WEIGHTS checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_all/model_reset_surgery.pth \
#                SOLVER.MAX_ITER 1000 SOLVER.CHECKPOINT_PERIOD 200 SOLVER.IMS_PER_BATCH_FEAT_EXTRACT 10 \
#                OUTPUT_DIR checkpoints_temp/coco_tfa_sgd/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_10shot_plus2

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/COCO-detection_TFA_SGD/faster_rcnn_R_101_FPN_ft_all_10shot.yaml \
#         --eval-all \
#         --opts OUTPUT_DIR checkpoints_temp/coco_tfa_sgd/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_10shot_plus2

python3 -m tools.train_net --num-gpus 1 \
        --config-file configs/COCO-detection_CG/faster_rcnn_R_101_FPN_ft_all_10shot.yaml \
        --opts MODEL.WEIGHTS checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_all/model_reset_surgery.pth \
               SOLVER.MAX_ITER 1000 SOLVER.CHECKPOINT_PERIOD 1001 \
               OUTPUT_DIR checkpoints_temp/coco_tfa_sgd_time/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_10shot
