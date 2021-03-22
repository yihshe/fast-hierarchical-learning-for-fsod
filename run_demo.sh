#!/bin/bash
#SBATCH  --output=sbatch_log/%j.out
#SBATCH  --gres=gpu:1 
#SBATCH  --mem=30G
source /scratch_net/biwidl13/yihshe/conda/etc/profile.d/conda.sh
conda activate pytcu10
python3 -m demo.demo \
--config-file configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_1shot.yaml \
--webcam \
--opts MODEL.WEIGHTS http://dl.yf.io/fs-det/models/coco/tfa_cos_1shot/model_final.pth