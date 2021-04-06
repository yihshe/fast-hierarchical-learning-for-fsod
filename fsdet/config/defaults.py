from detectron2.config import CfgNode as CN
from detectron2.config.defaults import _C

# adding additional default values built on top of the default values in detectron2

_CC = _C

# FREEZE Parameters
_CC.MODEL.BACKBONE.FREEZE = False
_CC.MODEL.PROPOSAL_GENERATOR.FREEZE = False
_CC.MODEL.ROI_HEADS.FREEZE_FEAT = False

# choose from "FastRCNNOutputLayers" and "CosineSimOutputLayers"
_CC.MODEL.ROI_HEADS.OUTPUT_LAYER = "FastRCNNOutputLayers"
# scale of cosine similarity (set to -1 for learnable scale)
_CC.MODEL.ROI_HEADS.COSINE_SCALE = 20.0

# set the path for the pretrained base model weights
_CC.MODEL.PRETRAINED_BASE_MODEL = None

# Backward Compatible options.
_CC.MUTE_HEADER = True

# Set the batch size for feature extraction
_CC.SOLVER.IMS_PER_BATCH_FEAT_EXTRACT = 10

# CG parameters
_CC.CG_PARAMS = CN()
_CC.CG_PARAMS.NUM_NEWTON_ITER = 100
_CC.CG_PARAMS.NUM_CG_ITER = 10
_CC.CG_PARAMS.INIT_HESSIAN_REG = 0.0
_CC.CG_PARAMS.HESSIAN_REG_FACTOR = 1.0
_CC.CG_PARAMS.CG_EPS = 0.0
_CC.CG_PARAMS.FLETCHER_REEVES = True
_CC.CG_PARAMS.STANDARD_ALPHA = True
_CC.CG_PARAMS.DIRECTION_FORGET_FACTOR = 0
_CC.CG_PARAMS.DEBUG = False
_CC.CG_PARAMS.ANALYZE_CONVERGENCE = False
_CC.CG_PARAMS.PLOTTING = False