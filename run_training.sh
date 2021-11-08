#!/bin/bash
#SBATCH  --output=sbatch_log/%j.out
#SBATCH  --gres=gpu:2
#SBATCH  --mem=30G

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
#         --two-stage-roi-heads \
#         --save-dir checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel_1shot_combine

# Random initialization
# python3 -m tools.ckpt_surgery \
#         --src1 checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1/model_final.pth \
#         --method randinit \
#         # --save-dir checkpoints_temp/voc/faster_rcnn/faster_rcnn_R_101_FPN_all2

# python3 -m tools.ckpt_surgery \
#         --src1 checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_hda_base/model_final.pth \
#         --method randinit \
#         --coco-new-setting

# python3 -m tools.ckpt_surgery \
#         --src1 checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_hda_base/model_final.pth \
#         --method randinit \
#         --save-dir checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_all \
#         --coco \
#         --two-stage-roi-heads-new-setting

# python3 -m tools.ckpt_surgery \
#         --src1 checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base3/model_final.pth \
#         --method randinit \
#         --two-stage-roi-heads

# python3 -m tools.ckpt_surgery \
#         --src1 checkpoints/lvis/faster_rcnn/faster_rcnn_R_101_FPN_base_norepeat_cos/model_final.pth \
#         --src2 checkpoints_temp/lvis/faster_rcnn/faster_rcnn_R_101_FPN_fc_novel_cosine_norepeat_hessian5e-1_iter1000/model_final.pth \
#         --method combine \
#         --lvis \
#         --two-stage-roi-heads

# Running SGD as same as the original implementation
# python3 -m tools.train_net --num-gpus 1 \
#         --config-file configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_1shot.yaml \
#         --opts MODEL.WEIGHTS checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel_1shot_combine/model_reset_combine.pth

# Running SGD after feature extraction, which can share same config file with CG
# python3 -m tools.train_net_modified --num-gpus 1 \
#         --config-file configs/COCO-detection_CG/faster_rcnn_R_101_FPN_ft_all_1shot.yaml \
#         # --opts MODEL.WEIGHTS checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel_1shot_combine/model_reset_combine.pth

# python3 -m tools.train_net_modified --num-gpus 1 \
#         --config-file configs/COCO-detection_CG/faster_rcnn_R_101_FPN_ft_novel_1shot.yaml \
#         --opts MODEL.WEIGHTS checkpoints_temp/CG_L2_RTS/coco/cg2_shot1_novel_iter20_metaweight_test/faster_rcnn/faster_rcnn_R_101_FPN_ft_novel_1shot/model_init.pth

# python3 -m tools.train_net_modified --num-gpus 1 \
#         --config-file configs/PascalVOC-detection_CG/split1/faster_rcnn_R_101_FPN_ft_all1_1shot.yaml \
#         --opts MODEL.WEIGHTS checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_all1/model_reset_surgery.pth

# python3 -m tools.train_net_modified_hda_new_setting --num-gpus 1 \
#         --config-file configs/COCO-detection_HDA_SGD_new_setting/faster_rcnn_R_101_FPN_ft_all_1shot.yaml

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/COCO-detection_HDA_SGD_new_setting/faster_rcnn_R_101_FPN_ft_all_1shot.yaml \
#         --eval-all

# python3 -m tools.train_net --num-gpus 2 \
#         --config-file configs/LVIS-detection/faster_rcnn_R_101_FPN_cosine_novel.yaml \
#         --opts MODEL.WEIGHTS checkpoints/lvis/faster_rcnn/faster_rcnn_R_101_FPN_base_repeat_cos/model_reset_remove.pth SOLVER.CHECKPOINT_PERIOD 10000

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/LVIS-detection/faster_rcnn_R_101_FPN_cosine_novel_norepeat.yaml \
#         --eval-only --eval-iter 20000

# python3 -m tools.ckpt_surgery \
#         --src1 checkpoints/lvis/faster_rcnn/faster_rcnn_R_101_FPN_base_repeat_cos/model_final.pth \
#         --method remove --lvis

# python3 -m tools.tune_net --num-gpus 1\
#         --config-file configs/COCO-detection_SCG/faster_rcnn_R_101_FPN_ft_all_10shot.yaml \
#         --SCGTrainer \
#         --opts SOLVER.MAX_ITER 200 SOLVER.CHECKPOINT_PERIOD 20 SOLVER.IMS_PER_BATCH 10 SOLVER.NUM_BATCHES_PER_SET 3 \
#                CG_PARAMS.NUM_NEWTON_ITER 1 \
#                OUTPUT_DIR checkpoints_temp/COCO_TFA_SCG/coco/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_10shot_SCG_test_newton1

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/COCO-detection_SCG/faster_rcnn_R_101_FPN_ft_all_10shot.yaml \
#         --eval-all \
#         --opts OUTPUT_DIR checkpoints_temp/COCO_TFA_SCG/coco/faster_rcnn/faster_rcnn_R_101_FPN_ft_all_10shot_SCG_test_newton1

python3 -m tools.train_net --num-gpus 2 \
        --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_3shot.yaml \
        --opts SOLVER.CHECKPOINT_PERIOD 12001 OUTPUT_DIR checkpoints_temp/voc_tfa_sgd_original/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_3shot_randnovel_l1

python3 -m tools.train_net --num-gpus 2 \
        --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_10shot.yaml \
        --opts SOLVER.CHECKPOINT_PERIOD 40001 OUTPUT_DIR checkpoints_temp/voc_tfa_sgd_original/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_10shot_randnovel_l1

# python3 -m tools.test_net --num-gpus 1 \
#         --config-file configs/COCO-detection_SCG/faster_rcnn_R_101_FPN_ft_all_10shot.yaml \
#         --eval-only \
#         --opts MODEL.WEIGHTS checkpoints/voc/faster_rcnn/faster_rcnn_R_101_FPN_base1/model_reset_surgery.pth \
#                OUTPUT_DIR checkpoints_temp/voc_tfa_sgd_original/faster_rcnn/faster_rcnn_R_101_FPN_ft_normalized_all1_5shot_randnovel_test


