from .inference import CGTrainer
from .inference_hda import HDACGTrainer
from .inference_stochastic import SCGTrainer
from .optimization import DetectionLossProblem, DetectionNewtonCG, MetaDetectionNewtonCG
from .gradient_mask import GradientMask
from .weight_predictor import WeightPredictor, WeightPredictor2, WeightPredictor3
from .feature_projector import FeatureProjector
