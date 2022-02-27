#!/bin/bash
#SBATCH  --output=sbatch_log/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=30G

source /scratch_net/biwidl13/yihshe/conda/etc/profile.d/conda.sh
conda activate pytcu10

# COCO 
# HDA hessian no feat_aug
# python3 -m tools.aggregate_seeds_CG --shots 1 --seeds 10 --coco \
#         --ckpt-path-prefix checkpoints/thesis/checkpoints_HDA_CG_new_setting --fc \
#         --print \
#         --new-setting

# python3 -m tools.aggregate_seeds_CG --shots 2 --seeds 10 --coco \
#         --ckpt-path-prefix checkpoints/thesis/checkpoints_HDA_CG_new_setting --fc \
#         --print \
#         --new-setting

# python3 -m tools.aggregate_seeds_CG --shots 3 --seeds 10 --coco \
#         --ckpt-path-prefix checkpoints/thesis/checkpoints_HDA_CG_new_setting --fc \
#         --print \
#         --new-setting

# python3 -m tools.aggregate_seeds_CG --shots 5 --seeds 10 --coco \
#         --ckpt-path-prefix checkpoints/thesis/checkpoints_HDA_CG_new_setting --fc \
#         --print \
#         --new-setting

# python3 -m tools.aggregate_seeds_CG --shots 10 --seeds 10 --coco \
#         --ckpt-path-prefix checkpoints/thesis/checkpoints_HDA_CG_new_setting --fc \
#         --print \
#         --new-setting

# Standard setting
# HDA
# archived_checkpoints/thesis/checkpoints_HDA_CG_augmentation
# HDA+SCG
# checkpoints/thesis/checkpoints_HDA_SCG
# HDA-wo-Aug
# archived_checkpoints/thesis/checkpoints_HDA_CG
# TFA-Fast
# archived_checkpoints/thesis/checkpoints_TFA_CG
# TFA*
# archived_checkpoints/thesis/checkpoints_TFA_SGD

# Novel seeting
# HDA
# archived_checkpoints/thesis/checkpoints_HDA_CG_new_setting_augmentation
# HDA-wo-Aug
# archived_checkpoints/thesis/checkpoints_HDA_CG_new_setting
# TFA
# archived_checkpoints/thesis/checkpoints_TFA_SGD_new_setting

python3 -m tools.aggregate_seeds_CG --shots 1 --seeds 10 --coco --fc --new-setting \
        --ckpt-path-prefix archived_checkpoints/thesis/checkpoints_HDA_CG_new_setting \
        --print

python3 -m tools.aggregate_seeds_CG --shots 2 --seeds 10 --coco --fc --new-setting \
        --ckpt-path-prefix archived_checkpoints/thesis/checkpoints_HDA_CG_new_setting \
        --print

python3 -m tools.aggregate_seeds_CG --shots 3 --seeds 10 --coco --fc --new-setting \
        --ckpt-path-prefix archived_checkpoints/thesis/checkpoints_HDA_CG_new_setting \
        --print

python3 -m tools.aggregate_seeds_CG --shots 5 --seeds 10 --coco --fc --new-setting \
        --ckpt-path-prefix archived_checkpoints/thesis/checkpoints_HDA_CG_new_setting \
        --print

python3 -m tools.aggregate_seeds_CG --shots 10 --seeds 10 --coco --fc --new-setting \
        --ckpt-path-prefix archived_checkpoints/thesis/checkpoints_HDA_CG_new_setting \
        --print



