"""Implement ROI_heads."""
import numpy as np
import torch
from torch import nn

import logging
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone.resnet import BottleneckBlock, make_stage
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.sampling import subsample_labels
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from detectron2.layers import batched_nms, cat
from typing import Dict
from torch.nn import functional as F

from .box_head import build_box_head
from .fast_rcnn import ROI_HEADS_OUTPUT_REGISTRY, FastRCNNOutputLayers, FastRCNNOutputs
from fsdet.data.builtin_meta import _get_builtin_metadata
from fsdet.data import HDAMetaInfo
from fsdet.data.lvis_v0_5_categories import LVIS_CATEGORIES, LVIS_CATEGORIES_NOVEL

from IPython import embed


ROI_HEADS_REGISTRY = Registry("ROI_HEADS")
ROI_HEADS_REGISTRY.__doc__ = """
Registry for ROI heads in a generalized R-CNN model.
ROIHeads take feature maps and region proposals, and
perform per-region computation.

The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`ROIHeads`.
"""

logger = logging.getLogger(__name__)


def build_roi_heads(cfg, input_shape):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.ROI_HEADS.NAME
    return ROI_HEADS_REGISTRY.get(name)(cfg, input_shape)


def select_foreground_proposals(proposals, bg_label):
    """
    Given a list of N Instances (for N images), each containing a `gt_classes` field,
    return a list of Instances that contain only instances with `gt_classes != -1 &&
    gt_classes != bg_label`.

    Args:
        proposals (list[Instances]): A list of N Instances, where N is the number of
            images in the batch.
        bg_label: label index of background class.

    Returns:
        list[Instances]: N Instances, each contains only the selected foreground instances.
        list[Tensor]: N boolean vector, correspond to the selection mask of
            each Instances object. True for selected instances.
    """
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    fg_proposals = []
    fg_selection_masks = []
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes
        fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label)
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        fg_proposals.append(proposals_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)
    return fg_proposals, fg_selection_masks


class ROIHeads(torch.nn.Module):
    """
    ROIHeads perform all per-region computation in an R-CNN.

    It contains logic of cropping the regions, extract per-region features,
    and make per-region predictions.

    It can have many variants, implemented as subclasses of this class.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super(ROIHeads, self).__init__()

        # fmt: off
        self.batch_size_per_image     = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        self.positive_sample_fraction = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
        self.test_score_thresh        = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        self.test_nms_thresh          = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        self.test_detections_per_img  = cfg.TEST.DETECTIONS_PER_IMAGE
        self.in_features              = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes              = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.proposal_append_gt       = cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT
        self.feature_strides          = {k: v.stride for k, v in input_shape.items()}
        self.feature_channels         = {k: v.channels for k, v in input_shape.items()}
        self.cls_agnostic_bbox_reg    = cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
        self.smooth_l1_beta           = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA
        # fmt: on

        # Matcher to assign box proposals to gt boxes
        self.proposal_matcher = Matcher(
            cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
            cfg.MODEL.ROI_HEADS.IOU_LABELS,
            allow_low_quality_matches=False,
        )

        # Box2BoxTransform for bounding box regression
        self.box2box_transform = Box2BoxTransform(
            weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS
        )

    def _sample_proposals(self, matched_idxs, matched_labels, gt_classes):
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes,
            self.batch_size_per_image,
            self.positive_sample_fraction,
            self.num_classes,
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(self, proposals, targets):
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns `self.batch_size_per_image` random samples from proposals and groundtruth boxes,
        with a fraction of positives that is no larger than `self.positive_sample_fraction.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:
                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                   then the ground-truth box is random)
                Other fields such as "gt_classes" that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(
                match_quality_matrix
            )
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # will filter the proposals again (by foreground/background,
                # etc), so we essentially index the data twice.
                for (
                    trg_name,
                    trg_value,
                ) in targets_per_image.get_fields().items():
                    if trg_name.startswith(
                        "gt_"
                    ) and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(
                            trg_name, trg_value[sampled_targets]
                        )
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros(
                        (len(sampled_idxs), 4)
                    )
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append(
                (gt_classes == self.num_classes).sum().item()
            )
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        # NOTE annotated for the test with gt
        # storage = get_event_storage()
        # storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        # storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

    def forward(self, images, features, proposals, targets=None):
        """
        Args:
            images (ImageList):
            features (dict[str: Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            proposals (list[Instances]): length `N` list of `Instances`s. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".
            targets (list[Instances], optional): length `N` list of `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
                It may have the following fields:
                - gt_boxes: the bounding box of each instance.
                - gt_classes: the label for each instance with a category ranging in [0, #class].

        Returns:
            results (list[Instances]): length `N` list of `Instances`s containing the
                detected instances. Returned during inference only; may be []
                during training.
            losses (dict[str: Tensor]): mapping from a named loss to a tensor
                storing the loss. Used during training only.
        """
        raise NotImplementedError()


@ROI_HEADS_REGISTRY.register()
class Res5ROIHeads(ROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where the heads share the
    cropping and the per-region feature computation by a Res5 block.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

        assert len(self.in_features) == 1

        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales     = (1.0 / self.feature_strides[self.in_features[0]], )
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON

        self.pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        self.res5, out_channels = self._build_res5_block(cfg)
        output_layer = cfg.MODEL.ROI_HEADS.OUTPUT_LAYER
        self.box_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(output_layer)(
            cfg, out_channels, self.num_classes, self.cls_agnostic_bbox_reg
        )

    def _build_res5_block(self, cfg):
        # fmt: off
        stage_channel_factor = 2 ** 3  # res5 is 8x res2
        num_groups           = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group      = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels  = num_groups * width_per_group * stage_channel_factor
        out_channels         = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        stride_in_1x1        = cfg.MODEL.RESNETS.STRIDE_IN_1X1
        norm                 = cfg.MODEL.RESNETS.NORM
        assert not cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE[-1], \
            "Deformable conv is not yet supported in res5 head."
        # fmt: on

        blocks = make_stage(
            BottleneckBlock,
            3,
            first_stride=2,
            in_channels=out_channels // 2,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            norm=norm,
            stride_in_1x1=stride_in_1x1,
        )
        return nn.Sequential(*blocks), out_channels

    def _shared_roi_transform(self, features, boxes):
        x = self.pooler(features, boxes)
        return self.res5(x)

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        del images

        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        pred_class_logits, pred_proposal_deltas = self.box_predictor(
            feature_pooled
        )
        del feature_pooled

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )

        if self.training:
            del features
            losses = outputs.losses()
            return [], losses
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_detections_per_img,
            )
            return pred_instances, {}


@ROI_HEADS_REGISTRY.register()
class StandardROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    The cropped rois go to separate branches directly.
    This way, it is easier to make separate abstractions for different branches.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    def __init__(self, cfg, input_shape):
        super(StandardROIHeads, self).__init__(cfg, input_shape)
        self._init_box_head(cfg)

    def _init_box_head(self, cfg):
        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [self.feature_channels[f] for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.box_head = build_box_head(
            cfg,
            ShapeSpec(
                channels=in_channels,
                height=pooler_resolution,
                width=pooler_resolution,
            ),
        )
        output_layer = cfg.MODEL.ROI_HEADS.OUTPUT_LAYER
        self.box_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(output_layer)(
            cfg,
            self.box_head.output_size,
            self.num_classes,
            self.cls_agnostic_bbox_reg,
        )

        self.with_gt  = False

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        # print('forward')
        # embed()
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)

        elif targets is not None:
            # NOTE maybe the step of sampling is not necessary for inference
            proposals = self.label_and_sample_proposals(proposals, targets)
            self.with_gt = True

        del targets

        features_list = [features[f] for f in self.in_features]

        if self.training:
            losses = self._forward_box(features_list, proposals)
            return proposals, losses
        else:
            pred_instances = self._forward_box(features_list, proposals)
            return pred_instances, {}

    def _forward_box(self, features, proposals):
        """
        Forward logic of the box prediction branch.

        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        box_features = self.box_pooler(
            features, [x.proposal_boxes for x in proposals]
        )
        box_features = self.box_head(box_features)
        pred_class_logits, pred_proposal_deltas = self.box_predictor(
            box_features
        )
        del box_features

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )
        if self.training:
            return outputs.losses()
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_detections_per_img,
                with_gt = self.with_gt,
            )
            return pred_instances
    
    # TODO
    # check the output
    # check the output of the proposal
    def extract_features(self, images, features, proposals, targets=None, extract_gt_box_features = False):
        """
        Extract features from the input data
        """
        del images
        assert self.training, "Model was changed to eval mode!"
        proposals = self.label_and_sample_proposals(proposals, targets)

        features_list = [features[f] for f in self.in_features]
        box_features = self._extract_features_box(features_list, proposals)
        if not extract_gt_box_features:
            del targets
            return proposals, box_features
        else:
            gt_box_features, gt_classes = self._extract_gt_features_box(features_list, targets)
            del targets
            return proposals, box_features, gt_box_features, gt_classes

    def _extract_features_box(self, features, proposals):
        """
        Forward logic of the box prediction branch for extracting features.

        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, extracted features (list[Tensor]).
        """
        # TODO replace proposals with targets and x.gt_boxes, one x corresponds to one img, also store gt.classes
        assert self.training, "Model was changed to eval mode!"
        box_features = self.box_pooler(
            features, [x.proposal_boxes for x in proposals]
        )
        box_features = self.box_head(box_features)


        return box_features

    # TODO filter the novel classes and only extract the novel classes
    def _extract_gt_features_box(self, features, targets):
        """
        Forward logic of the box prediction branch for extracting features for ground truth bbox.

        Args:
            features (list[Tensor]): #level input features for box prediction
            targets (list[Instances]): the per-image ground truth of Instances.
                Each has fields "gt_classes", "gt_boxes".

        Returns:
            In training, extracted features (list[Tensor]), gt_classes (list[Tensor])
        """
        assert self.training, "Model was changed to eval mode!"                
        # extract features for all ground truth boxes
        gt_box_features = self.box_pooler(
            features, [x.gt_boxes for x in targets]
        )
        gt_box_features = self.box_head(gt_box_features)
        
        gt_classes = [x.gt_classes for x in targets]
        
        return gt_box_features, gt_classes

    def losses_from_features(self, box_features, proposals, weights = None, super_cat: str = None):
        """
        Forward logic of the box prediction branch for computing losses.

        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
        """
        assert self.training, "Model was changed to eval mode!"
        pred_class_logits, pred_proposal_deltas = self.box_predictor(
            box_features, weights
        )
        del box_features

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )
        return outputs.losses()

# NOTE HDAROIHeads for standard setting. this ROI head applies hierarchical approach to detect classes in TFA setting (only super class of background)
@ROI_HEADS_REGISTRY.register()
class TwoStageROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    The cropped rois go to separate branches directly.
    This way, it is easier to make separate abstractions for different branches.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    def __init__(self, cfg, input_shape):
        super(TwoStageROIHeads, self).__init__(cfg, input_shape)
        self._init_box_head(cfg)

    def _init_box_head(self, cfg):
        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [self.feature_channels[f] for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.box_head = build_box_head(
            cfg,
            ShapeSpec(
                channels=in_channels,
                height=pooler_resolution,
                width=pooler_resolution,
            ),
        )
        output_layer = cfg.MODEL.ROI_HEADS.OUTPUT_LAYER

        # TODO LVIS modify the following params in the config accordingly
        # NOTE in LVIS training, we trainn it directly on the Rare dataset, so num_classes is 454 when label the bg id
        self.lvis = True if 'lvis' in cfg.DATASETS.TRAIN[0] else False

        # TODO TODO fix the bug and test the code
        # if self.training and self.lvis:
        #     self.num_classes  = cfg.MODEL.ROI_HEADS.NUM_CLASSES_NOVEL
        
        self.num_classes_base = cfg.MODEL.ROI_HEADS.NUM_CLASSES_BASE
        self.num_classes_novel = cfg.MODEL.ROI_HEADS.NUM_CLASSES_NOVEL
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        metadata = self.get_metadata(cfg.DATASETS.TRAIN[0])
        # metadata = _get_builtin_metadata('coco_fewshot')
        # TODO lvis c, adjust the idmap here
        self.idmaps = self.init_idmaps(metadata)
        self.metadata = metadata

        self.box_predictor_base = ROI_HEADS_OUTPUT_REGISTRY.get(output_layer)(
            cfg,
            self.box_head.output_size,
            # NOTE HDA num of base classes should be 42 for the first hier level (also related to the weight initialization)
            self.num_classes_base,
            self.cls_agnostic_bbox_reg,
        )
        # train it using CG
        self.box_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(output_layer)(
            cfg,
            self.box_head.output_size,
            self.num_classes_novel,
            self.cls_agnostic_bbox_reg,
        )
        # NOTE HDA add two more roi heads for the child classes of animal and food
        self.with_gt = False
        self.alternative_test_score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST_SECOND

    def get_metadata(self, dataset_name):
        metadata = dict({})
        # NOTE HDA the metadata for coco to get is coco_fewshot_base_all
        # NOTE LVIS add the meta data of lvis needed for proposal filtering
        if 'coco' in dataset_name:
            metadata = _get_builtin_metadata('coco_fewshot')
        elif 'voc' in dataset_name:
            sid = int(dataset_name.split('all')[1].split('_')[0])
            init_metadata = _get_builtin_metadata('pascal_voc_fewshot')
            metadata['thing_classes'] = init_metadata['thing_classes'][sid]
            metadata['base_classes'] = init_metadata['base_classes'][sid]
            metadata['novel_classes'] = init_metadata['novel_classes'][sid]
            metadata['thing_dataset_id_to_contiguous_id'] = {i:i for i,v in enumerate(metadata['thing_classes'])}
            metadata['base_dataset_id_to_contiguous_id'] = {i:i for i,v in enumerate(metadata['base_classes'])}
            metadata['novel_dataset_id_to_contiguous_id'] = {len(metadata['base_classes'])+i:i for i,v in enumerate(metadata['novel_classes'])}
        elif 'lvis' in dataset_name:
            # NOTE the metadata used here should contain thing, base, novel, which coulld be different to the metadata of the dataset
            # if the novel detector is trained directly on the Rare dataset, then the metadata here will only be used in inference.
            assert len(LVIS_CATEGORIES) == 1230
            cat_ids = [k["id"] for k in LVIS_CATEGORIES]
            assert min(cat_ids) == 1 and max(cat_ids) == len(cat_ids), "Category ids are not in [1, #categories], as expected"
            
            # Ensure that the category list is sorted by id
            lvis_categories = [k for k in sorted(LVIS_CATEGORIES, key=lambda x: x["id"])]
            thing_classes = [k["synonyms"][0] for k in lvis_categories]
            
            lvis_categories_novel = [k for k in sorted(LVIS_CATEGORIES_NOVEL, key=lambda x: x["id"])] 
            novel_classes = [k["synonyms"][0] for k in lvis_categories_novel]
            
            lvis_categories_base = [k for k in sorted(LVIS_CATEGORIES, key=lambda x: x["id"]) if k["synonyms"][0] not in novel_classes]
            base_classes = [k["synonyms"][0] for k in lvis_categories_base]


            metadata["thing_classes"]= thing_classes
            metadata["thing_dataset_id_to_contiguous_id"] = {x["id"]:i for i,x in enumerate(lvis_categories)}
            metadata["novel_classes"] = novel_classes
            metadata["novel_dataset_id_to_contiguous_id"] = {x["id"]:i for i,x in enumerate(lvis_categories_novel)}
            metadata["base_classes"] = base_classes
            metadata["base_dataset_id_to_contiguous_id"] = {x["id"]:i for i,x in enumerate(lvis_categories_base)}
            
        return metadata

    # TODO LVIS new init_idmaps for LVIS or put it in a separate class
    def init_idmaps(self, metadata):
        idmap_global = metadata['thing_dataset_id_to_contiguous_id']
        idmap_global_reversed = {v: k for k, v in idmap_global.items()}

        idmap_base = metadata['base_dataset_id_to_contiguous_id']
        idmap_base_reversed = {v: k for k, v in idmap_base.items()}
        base_class_ids_global = [idmap_global[k] for k in idmap_base.keys()]

        idmap_novel = metadata['novel_dataset_id_to_contiguous_id']
        idmap_novel_reversed = {v: k for k, v in idmap_novel.items()}
        novel_class_ids_global = [idmap_global[k] for k in idmap_novel.keys()]

        return {
            'idmap_global': idmap_global,
            'idmap_global_reversed': idmap_global_reversed,
            'idmap_base': idmap_base,
            'idmap_base_reversed': idmap_base_reversed,
            'base_class_ids_global': base_class_ids_global,
            'idmap_novel': idmap_novel,
            'idmap_novel_reversed': idmap_novel_reversed,
            'novel_class_ids_global': novel_class_ids_global,
        }
        
    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        # NOTE gt
        elif targets is not None:
            proposals = self.label_and_sample_proposals(proposals, targets)
            self.with_gt = True
        
        del targets

        features_list = [features[f] for f in self.in_features]
        
        if self.training:
            losses = self._forward_box(features_list, proposals)
            return proposals, losses
        else:
            pred_instances = self._forward_box(features_list, proposals)
            return pred_instances, {}
    
    def _forward_box(self, features, proposals):
        """
        Forward logic of the box prediction branch.

        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        box_features = self.box_pooler(
            features, [x.proposal_boxes for x in proposals]
        )
        box_features = self.box_head(box_features)

        # NOTE If the dataset is LVIS and it is in training mode, then filtering is not needed as train the model directly on Rare dataset
        # Overall the forward_box function is only used for batch case, for both training and inference
        # actually currently this function only support the training for LVIS dataset, in SCG case (also SGD in the future)
        
        # if self.training and self.lvis:
        #     pred_class_logits_novel, pred_proposal_deltas_novel = self.box_predictor(box_features)
        #     proposals_novel = proposals
        # else:
        pred_class_logits_base, pred_proposal_deltas_base = self.box_predictor_base(box_features)
        pred_instances_base, bg_inds, proposals_novel = self.detection_filter(proposals, pred_class_logits_base, pred_proposal_deltas_base)
        
        # selected base proposals, and corresponding logits and deltas 
        # pred_class_logits_base, pred_proposal_deltas_base = pred_class_logits_base[fg_inds], pred_proposal_deltas_base[fg_inds]
        pred_class_logits_novel, pred_proposal_deltas_novel = self.box_predictor(box_features[bg_inds])
        
        del box_features

        # NOTE gt
        outputs_novel = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits_novel,
            pred_proposal_deltas_novel,
            proposals_novel,
            self.smooth_l1_beta,
        )

        if self.training:
            return outputs_novel.losses()
        else:
            pred_instances_novel, _ = outputs_novel.inference(
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_detections_per_img,
                with_gt = self.with_gt,
            )
            pred_instances = self.merge_instances(pred_instances_base, pred_instances_novel)
        
            return pred_instances
  
    def merge_instances(self, pred_instances_base, pred_instances_novel):
        idmap_global = self.idmaps['idmap_global']
        idmap_base_reversed = self.idmaps['idmap_base_reversed']
        idmap_novel_reversed = self.idmaps['idmap_novel_reversed']

        pred_instances = list([])
        for instance_base, instance_novel in zip(pred_instances_base, pred_instances_novel):
            assert instance_base.image_size == instance_novel.image_size
            instance = Instances(instance_base.image_size)
            instance.pred_boxes = Boxes(torch.cat((instance_base.pred_boxes.tensor, instance_novel.pred_boxes.tensor)))
            instance.scores = torch.cat((instance_base.scores, instance_novel.scores))

            pred_classes_mapped_base = torch.zeros(instance_base.pred_classes.shape).to(self.device)
            for i, pred_class in enumerate(instance_base.pred_classes.cpu().numpy()):
                pred_classes_mapped_base[i] = idmap_global[idmap_base_reversed[pred_class]]

            pred_classes_mapped_novel = torch.zeros(instance_novel.pred_classes.shape).to(self.device)
            for i, pred_class in enumerate(instance_novel.pred_classes.cpu().numpy()):
                pred_classes_mapped_novel[i] = idmap_global[idmap_novel_reversed[pred_class]]
            
            instance.pred_classes = torch.cat((pred_classes_mapped_base, pred_classes_mapped_novel)).int()
            # NOTE gt
            if instance_base.has("gt_classes"):
                instance.gt_classes = torch.cat((instance_base.gt_classes, instance_novel.gt_classes))
            pred_instances.append(instance)
    
        return pred_instances

    def detection_filter(self, proposals, pred_class_logits_base, pred_proposal_deltas_base):
        # NOTE for test
        outputs_base_simulator = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits_base,
            pred_proposal_deltas_base,
            proposals,
            self.smooth_l1_beta,
        )

        # NOTE gt
        pred_instances_fg, fg_inds_local_list = outputs_base_simulator.inference(
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_detections_per_img,
            with_gt = self.with_gt,
            alternative_score_thresh = self.alternative_test_score_thresh,
        )
        # embed()
        
        num_preds_per_image = [len(p) for p in proposals]
        # fg_inds = torch.tensor([]).to(self.device)
        bg_inds = torch.tensor([]).to(self.device)
        # NOTE HDA proposals_base will also be needed in training
        # proposals_base = None if self.training else list([]) 
        proposals_novel = list([])
        
        pred_classes = pred_class_logits_base.argmax(dim=1)
        bg_class_id = pred_class_logits_base.shape[1]-1
        bg_score_max_bool_list = (pred_classes == bg_class_id).split(num_preds_per_image, dim=0)
        
        # NOTE VOC filtering test
        # scores = F.softmax(pred_class_logits_base,dim=-1)
        # filter_mask = scores > 0.5
        # filter_inds = filter_mask.nonzero()
        # obj_inds = set(filter_inds[:,0].unique().cpu().tolist())
        # bg_score_max_bool_list = torch.tensor([i in obj_inds for i in range(pred_class_logits_base.shape[0])]).split(num_preds_per_image, dim=0)

        num_preds_per_image = torch.tensor(num_preds_per_image).long()
        
        # NOTE HDA here we filter the proposals for animal, food, and background
        for i, (fg_inds_local, proposal, num_preds, bg_score_max_bool) in enumerate(zip(fg_inds_local_list, proposals, num_preds_per_image, bg_score_max_bool_list)):

            # fg_inds_local, _ = fg_inds_local.long().unique().sort()
            # bg_inds_local = torch.tensor([i for i in torch.arange(num_preds).to(self.device)[bg_score_max_bool] if i not in fg_inds_local]).long().to(self.device)
            fg_inds_local = set(fg_inds_local.unique().cpu().tolist())
            bg_inds_local = torch.tensor([i for i in torch.arange(num_preds)[bg_score_max_bool.cpu()] if i.item() not in fg_inds_local]).long().to(self.device)
        
            # given the pred_classes, select the fg_inds and then filter the fg_inds_local, and concatenate to filter the features for prediction
            # fg_inds = torch.cat((fg_inds, torch.sum(num_preds_per_image[:i])+fg_inds_local)).long()
            bg_inds = torch.cat((bg_inds, torch.sum(num_preds_per_image[:i])+bg_inds_local)).long()
            
            # NOTE HDA for training, we also need to keep the proposals_base to filter the proposals given the pred_classes
            # if proposals_base is not None:
            #     proposals_base.append(self.proposal_filter(proposal, fg_inds_local, novel = False))

            proposals_novel.append(self.proposal_filter(proposal, bg_inds_local, novel = True))

        # return pred_instances_fg, fg_inds, bg_inds, proposals_base, proposals_novel
        return pred_instances_fg, bg_inds, proposals_novel

    def proposal_filter(self, proposal, inds, novel = False):
        proposal_filtered = Instances(proposal.image_size)
        proposal_filtered.proposal_boxes = Boxes(proposal.proposal_boxes.tensor[inds])
        proposal_filtered.objectness_logits = proposal.objectness_logits[inds]
        
        # only for training we have the gt_classes
        if proposal.has('gt_boxes'):
            proposal_filtered.gt_boxes = Boxes(proposal.gt_boxes.tensor[inds])

            assert proposal.has('gt_classes')
            
            if self.with_gt == True:
                # If with_gt is True, it means that it is in test phase, rather than training phase
                proposal_filtered.gt_classes = proposal.gt_classes[inds]
            else:
                # NOTE add part for gt in test
                idmap_global = self.idmaps['idmap_global']
                idmap_global_reversed = self.idmaps['idmap_global_reversed']
                idmap_local = self.idmaps['idmap_novel'] if novel is True else self.idmaps['idmap_base']
                # NOTE HDA the num_classes_base here should be 60 child classes rather than 42
                bg_class_id_global = self.num_classes_base+self.num_classes_novel
                bg_class_id_local = self.num_classes_novel if novel is True else self.num_classes_base
                other_class_ids_global = self.idmaps['base_class_ids_global'] if novel is True else self.idmaps['novel_class_ids_global']
                other_class_ids_global = set(list(other_class_ids_global) + [bg_class_id_global])

                gt_classes_filtered = proposal.gt_classes[inds]
                gt_classes_mapped = torch.zeros(gt_classes_filtered.shape).to(self.device)
                for i, gt_class in enumerate(gt_classes_filtered.cpu().numpy()):
                    gt_classes_mapped[i] = bg_class_id_local if gt_class in other_class_ids_global else idmap_local[idmap_global_reversed[gt_class]]
                    # print('gt_class')
                    # embed()
                proposal_filtered.gt_classes = gt_classes_mapped.long()

        return proposal_filtered

    # TODO LVIS compile this function with batch SCG
    def extract_features(self, images, features, proposals, targets=None, extract_gt_box_features = False):
        """
        Extract features from the input data
        """
        del images
        assert self.training, "Model was changed to eval mode!"
        proposals = self.label_and_sample_proposals(proposals, targets)
        # del targets

        features_list = [features[f] for f in self.in_features]
        proposals_novel, box_features_novel, pred, true = self._extract_features_box(features_list, proposals)
        if not extract_gt_box_features:
            del targets
            return proposals_novel, box_features_novel, pred, true
        else:
            gt_box_features, gt_classes = self._extract_gt_features_box(features_list, targets)
            del targets
            return proposals_novel, box_features_novel, gt_box_features, gt_classes

    def _extract_features_box(self, features, proposals):
        """
        Forward logic of the box prediction branch for extracting features.

        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, extracted features (list[Tensor]).
        """
        # TODO replace proposals with targets and x.gt_boxes, one x corresponds to one img, also store gt.classes
        assert self.training, "Model was changed to eval mode!"
        box_features = self.box_pooler(
            features, [x.proposal_boxes for x in proposals]
        )
        box_features = self.box_head(box_features)
        
        # if self.lvis:
        #     # NOTE for LVIS, the model will be trained directly on novel dataset
        #     proposals_novel = proposals
        #     box_features_novel = box_features
        # else:
        pred_class_logits_base, pred_proposal_deltas_base = self.box_predictor_base(box_features)
        pred_instances_base, bg_inds, proposals_novel = self.detection_filter(proposals, pred_class_logits_base, pred_proposal_deltas_base)

        box_features_novel = box_features[bg_inds] if bg_inds.shape[0] != 0 else torch.tensor([]).to(self.device)
        
        # NOTE exp for rebuttal
        pred_classes = pred_class_logits_base.argmax(dim=1)
        true_classes = torch.cat(tuple(proposal.gt_classes for proposal in proposals))
        
        return proposals_novel, box_features_novel, pred_classes, true_classes

    # TODO filter the novel classes and only extract the novel classes
    def _extract_gt_features_box(self, features, targets):
        """
        Forward logic of the box prediction branch for extracting features for ground truth bbox.

        Args:
            features (list[Tensor]): #level input features for box prediction
            targets (list[Instances]): the per-image ground truth of Instances.
                Each has fields "gt_classes", "gt_boxes".

        Returns:
            In training, extracted features (list[Tensor]), gt_classes (list[Tensor])
        """
        # TODO this has not been adapted for the two stage training
        assert self.training, "Model was changed to eval mode!"                
        # extract features for all ground truth boxes
        gt_box_features = self.box_pooler(
            features, [x.gt_boxes for x in targets]
        )
        gt_box_features = self.box_head(gt_box_features)
        
        gt_classes = [x.gt_classes for x in targets]
        
        return gt_box_features, gt_classes

    def losses_from_features(self, box_features, proposals, weights = None, super_cat:str = None):
        """
        Forward logic of the box prediction branch for computing losses.

        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
        """
        assert self.training, "Model was changed to eval mode!"
       
        pred_class_logits_novel, pred_proposal_deltas_novel = self.box_predictor(
            box_features, weights
        )
        del box_features
        # print('loss')
        # embed()
        outputs_novel = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits_novel,
            pred_proposal_deltas_novel,
            proposals,
            self.smooth_l1_beta,
        )

        return outputs_novel.losses()

# NOTE HDAROIHeads for new setting
@ROI_HEADS_REGISTRY.register()
class HDAROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    The cropped rois go to separate branches directly.
    This way, it is easier to make separate abstractions for different branches.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    def __init__(self, cfg, input_shape):
        super(HDAROIHeads, self).__init__(cfg, input_shape)
        self._init_box_head(cfg)

    def _init_box_head(self, cfg):
        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [self.feature_channels[f] for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.box_head = build_box_head(
            cfg,
            ShapeSpec(
                channels=in_channels,
                height=pooler_resolution,
                width=pooler_resolution,
            ),
        )
        output_layer = cfg.MODEL.ROI_HEADS.OUTPUT_LAYER
        output_layer_base_super_cat = "HDABaseCatFastRCNNOutputLayers"

        # TODO num_classes_base_hier1 = 42, num_classes_base_hier2 = 60, num_classes
        # self.num_classes_base = cfg.MODEL.ROI_HEADS.NUM_CLASSES_BASE
        # self.num_classes_novel = cfg.MODEL.ROI_HEADS.NUM_CLASSES_NOVEL

        self.num_classes_dict = {
            'hier1_fg': cfg.MODEL.ROI_HEADS.NUM_CLASSES_HIER1,
            'hier2_fg': cfg.MODEL.ROI_HEADS.NUM_CLASSES_HIER2_FG,
            'hier2_bg': cfg.MODEL.ROI_HEADS.NUM_CLASSES_HIER2_BG,
            'hier2_animal': cfg.MODEL.ROI_HEADS.NUM_CLASSES_HIER2_FG_ANIMAL,
            'hier2_food': cfg.MODEL.ROI_HEADS.NUM_CLASSES_HIER2_FG_FOOD
        }
    
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # TODO modify the metadata function to return the needed data
        self.hda_meta_info = HDAMetaInfo()
        metadata = self.get_metadata(cfg.DATASETS.TRAIN[0])
        self.idmaps = self.init_idmaps(metadata, self.hda_meta_info)

        self.animal_class_id = self.idmaps['idmap_hier1_global'][self.hda_meta_info.base_model_cats_name2id['animal']]
        self.food_class_id = self.idmaps['idmap_hier1_global'][self.hda_meta_info.base_model_cats_name2id['food']]
        
        
        # Frozen base predictor 
        self.box_predictor_base = ROI_HEADS_OUTPUT_REGISTRY.get(output_layer)(
            cfg,
            self.box_head.output_size,
            # NOTE HDA num of base classes should be 42 for the first hier level (also related to the weight initialization)
            self.num_classes_dict['hier1_fg'],
            self.cls_agnostic_bbox_reg,
        )

        # Learnable predictor for background classes
        self.box_predictor_bg = ROI_HEADS_OUTPUT_REGISTRY.get(output_layer)(
            cfg,
            self.box_head.output_size,
            self.num_classes_dict['hier2_bg'],
            self.cls_agnostic_bbox_reg,
        )
        # NOTE TODO for the predictor of animal and food, no bg class needs to be predicted
        # NOTE TODO further processing of the predicted scores
        # Learnable predictor for superclass animal
        self.box_predictor_animal = ROI_HEADS_OUTPUT_REGISTRY.get(output_layer)(
            cfg,
            self.box_head.output_size,
            self.num_classes_dict['hier2_animal'],
            self.cls_agnostic_bbox_reg,
            super_base_class = True
        )
        # Learnable predictor for superclass food
        self.box_predictor_food = ROI_HEADS_OUTPUT_REGISTRY.get(output_layer)(
            cfg,
            self.box_head.output_size,
            self.num_classes_dict['hier2_food'],
            self.cls_agnostic_bbox_reg,
            super_base_class = True
        )

        self.iter_count = 0

        self.with_gt = False

    # TODO depending on what metadata is needed in the following section
    def get_metadata(self, dataset_name):
        # NOTE HDA the metadata for coco to get is coco_fewshot_hda_all
        # currently the task is only designed for dataset coco_hda_all
        if 'coco' in dataset_name:
            metadata = _get_builtin_metadata('coco_fewshot_hda_all')
        return metadata

    def init_idmaps(self, metadata, hda_meta_info: HDAMetaInfo):
        idmap_hier1_global = hda_meta_info.get_meta_hda_base()['base_dataset_id_to_contiguous_id']
        idmap_hier1_global_reversed = {v: k for k, v in idmap_hier1_global.items()}

        idmap_hier2_global = metadata['thing_dataset_id_to_contiguous_id']
        idmap_hier2_global_reversed = {v: k for k, v in idmap_hier2_global.items()}

        idmap_hier2_fg = metadata['base_dataset_id_to_contiguous_id']
        idmap_hier2_fg_reversed = {v: k for k, v in idmap_hier2_fg.items()}
        hier2_fg_class_ids_global = [idmap_hier2_global[k] for k in idmap_hier2_fg.keys()]

        idmap_hier2_bg = metadata['novel_dataset_id_to_contiguous_id']
        idmap_hier2_bg_reversed = {v: k for k, v in idmap_hier2_bg.items()}
        hier2_bg_class_ids_global = [idmap_hier2_global[k] for k in idmap_hier2_bg.keys()]

        idmap_hier2_animal = {child_cat: i for i, child_cat in enumerate(hda_meta_info.super_cats_to_child_cats_idmap[hda_meta_info.super_cats_name2id['animal']])}
        idmap_hier2_animal_reversed = {v: k for k, v in idmap_hier2_animal.items()}
        hier2_animal_class_ids_global = [idmap_hier2_global[k] for k in idmap_hier2_animal.keys()]

        idmap_hier2_food = {child_cat: i for i, child_cat in enumerate(hda_meta_info.super_cats_to_child_cats_idmap[hda_meta_info.super_cats_name2id['food']])}
        idmap_hier2_food_reversed = {v: k for k, v in idmap_hier2_food.items()}
        hier2_food_class_ids_global = [idmap_hier2_global[k] for k in idmap_hier2_food.keys()]

        return {
            'idmap_hier1_global': idmap_hier1_global,
            'idmap_hier1_global_reversed': idmap_hier1_global_reversed,

            'idmap_hier2_global': idmap_hier2_global,
            'idmap_hier2_global_reversed': idmap_hier2_global_reversed,

            'idmap_hier2_fg': idmap_hier2_fg,
            'idmap_hier2_fg_reversed': idmap_hier2_fg_reversed,
            'hier2_fg_class_ids_global': hier2_fg_class_ids_global,

            'idmap_hier2_bg': idmap_hier2_bg,
            'idmap_hier2_bg_reversed': idmap_hier2_bg_reversed,
            'hier2_bg_class_ids_global': hier2_bg_class_ids_global,

            'idmap_hier2_animal': idmap_hier2_animal,
            'idmap_hier2_animal_reversed': idmap_hier2_animal_reversed,
            'hier2_animal_class_ids_global': hier2_animal_class_ids_global,

            'idmap_hier2_food': idmap_hier2_food,
            'idmap_hier2_food_reversed': idmap_hier2_food_reversed,
            'hier2_food_class_ids_global': hier2_food_class_ids_global,
        }
        
    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        elif targets is not None:
            proposals = self.label_and_sample_proposals(proposals, targets)
            self.with_gt = True
        
        del targets

        features_list = [features[f] for f in self.in_features]

        # if self.training:
        #     losses = self._forward_box(features_list, proposals)
        #     return proposals, losses
        # else:
        #     pred_instances = self._forward_box(features_list, proposals)
        #     return pred_instances, {}
        assert not self.training, "Currently, forward function in HDAROIHeads only support inference!"
        pred_instances = self._forward_box(features_list, proposals)
        return pred_instances, {}

    # TODO modify this fuction after detection filter
    def _forward_box(self, features, proposals):
        """
        Forward logic of the box prediction branch.

        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        box_features = self.box_pooler(
            features, [x.proposal_boxes for x in proposals]
        )
        box_features = self.box_head(box_features)

        # NOTE box predictor should be frozen
        pred_class_logits_base, pred_proposal_deltas_base = self.box_predictor_base(box_features)
        # TODO TODO change the way to filter fg super class inds with gt labels if it is in training
        pred_instances_fg_fixed_cats, fg_animal_inds, fg_food_inds, bg_inds, proposals_animal, proposals_food, proposals_bg = self.detection_filter(
            proposals, pred_class_logits_base, pred_proposal_deltas_base,
        )
        
        # selected base proposals, and corresponding logits and deltas 
        # NOTE TODO the predictor of animal and food should only output the classification scores
        pred_class_logits_bg, pred_proposal_deltas_bg = self.box_predictor_bg(box_features[bg_inds])
        pred_class_logits_animal, pred_proposal_deltas_animal = self.box_predictor_animal(box_features[fg_animal_inds])
        pred_class_logits_food, pred_proposal_deltas_food = self.box_predictor_food(box_features[fg_food_inds])
        del box_features

        # TODO implement SGD training here (for batch features)

        # NOTE the _forward_box is now only used for inference 
        # assert not self.training
        assert not self.training, "Forward function in HDAROIHeads does not support training!"
        # NOTE currently in HDA new setting, 
        pred_instances_fg_fixed_cats = self.fg_child_cat_id_mapping(pred_instances_fg_fixed_cats)    
        pred_instances_animal = self.fg_super_cat_inference(pred_class_logits_animal, pred_proposal_deltas_animal, proposals_animal, super_cat='animal')
        pred_instances_food = self.fg_super_cat_inference(pred_class_logits_food, pred_proposal_deltas_food, proposals_food, super_cat='food')
        
        pred_instances_bg = self.bg_inference(pred_class_logits_bg, pred_proposal_deltas_bg, proposals_bg)

        pred_instances = self.merge_instances(pred_instances_fg_fixed_cats, pred_instances_animal, pred_instances_food, pred_instances_bg)
        
        return pred_instances

    def fg_child_cat_id_mapping(self, pred_instances_fg):
        idmap_global = self.idmaps['idmap_hier2_global']
        idmap_local_reversed = self.idmaps['idmap_hier1_global_reversed']

        for pred_instance_fg in pred_instances_fg:
            for i, pred_class in enumerate(pred_instance_fg.pred_classes.cpu().numpy()):
                assert pred_class not in [self.animal_class_id, self.food_class_id]
                pred_instance_fg.pred_classes[i] = idmap_global[idmap_local_reversed[pred_class]]

        return pred_instances_fg

    def fg_super_cat_inference(self, pred_class_logits, pred_proposal_deltas, proposals, super_cat: str):
        assert super_cat in ['animal', 'food']

        # TODO now, for the inference, look into what is the difference for the inference of super base classes. here we need nms
        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
            super_base_class=True,
        )
        # TODO NOTE should add with_gt?
        pred_instances_fg_super_cat, _ = outputs.inference(
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_detections_per_img,
        )

        idmap_global = self.idmaps['idmap_hier2_global']
        idmap_local_reversed = self.idmaps['idmap_hier2_{}_reversed'.format(super_cat)]
       
        for pred_instance_fg_super_cat in pred_instances_fg_super_cat:
            for i, pred_class in enumerate(pred_instance_fg_super_cat.pred_classes.cpu().numpy()):
                pred_instance_fg_super_cat.pred_classes[i] = idmap_global[idmap_local_reversed[pred_class]]
        
        return pred_instances_fg_super_cat

    def bg_inference(self, pred_class_logits_bg, pred_proposal_deltas_bg, proposals_bg):
        idmap_global = self.idmaps['idmap_hier2_global']
        idmap_local_reversed = self.idmaps['idmap_hier2_bg_reversed']
        outputs_bg = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits_bg,
            pred_proposal_deltas_bg,
            proposals_bg,
            self.smooth_l1_beta,
        )
        pred_instances_bg, _ = outputs_bg.inference(
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_detections_per_img,
            with_gt = self.with_gt,
        )

        for pred_instance_bg in pred_instances_bg:
            for i, pred_class in enumerate(pred_instance_bg.pred_classes.cpu().numpy()):
                pred_instance_bg.pred_classes[i] = idmap_global[idmap_local_reversed[pred_class]]
        
        return pred_instances_bg

    # TODO now, pred_instances_animal, pred_instances_food also included.
    def merge_instances(self, pred_instances_fg_fixed_cats, pred_instances_animal, pred_instances_food, pred_instances_bg):
        pred_instances = list([])
        for instance_fg_fixed_cats, instance_animal, instance_food, instance_bg in zip(pred_instances_fg_fixed_cats, pred_instances_animal, pred_instances_food, pred_instances_bg):
            instance = Instances(instance_fg_fixed_cats.image_size)
            instance.pred_boxes = Boxes(torch.cat((instance_fg_fixed_cats.pred_boxes.tensor, instance_animal.pred_boxes.tensor, instance_food.pred_boxes.tensor, instance_bg.pred_boxes.tensor)))
            instance.scores = torch.cat((instance_fg_fixed_cats.scores, instance_animal.scores, instance_food.scores, instance_bg.scores))
            instance.pred_classes = torch.cat((instance_fg_fixed_cats.pred_classes, instance_animal.pred_classes, instance_food.pred_classes, instance_bg.pred_classes))
            if instance_fg_fixed_cats.has("gt_classes"):
                instance.gt_classes = torch.cat((instance_fg_fixed_cats.gt_classes, instance_animal.gt_classes, instance_food.gt_classes, instance_bg.gt_classes))
            pred_instances.append(instance)
        return pred_instances

    def detection_filter(self, proposals, pred_class_logits_base, pred_proposal_deltas_base):
        outputs_base_simulator = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits_base,
            pred_proposal_deltas_base,
            proposals,
            self.smooth_l1_beta,
        )

        pred_instances_fg, fg_inds_local_list = outputs_base_simulator.inference(
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_detections_per_img,
            with_gt = self.with_gt,
        )
        
        pred_instances_fg_fixed_cats = list([])
        num_preds_per_image = [len(p) for p in proposals]
        # fg_inds = torch.tensor([]).to(self.device)
        fg_animal_inds = torch.tensor([]).to(self.device)
        fg_food_inds = torch.tensor([]).to(self.device)
        bg_inds = torch.tensor([]).to(self.device)

        # NOTE HDA proposals_base will also be needed in training
        # proposals_base = None if self.training else list([]) 
        proposals_animal = list([])
        proposals_food = list([])
        proposals_bg = list([])
        # proposals_novel = list([])
        
        # pred_classes = pred_class_logits_base.argmax(dim=1)
        bg_class_id = pred_class_logits_base.shape[1]-1
        bg_score_max_bool_list = (pred_class_logits_base.argmax(dim=-1) == bg_class_id).split(num_preds_per_image, dim=0)
        animal_score_max_bool_list = (pred_class_logits_base.argmax(dim=-1) == self.animal_class_id).split(num_preds_per_image, dim=0)
        food_score_max_bool_list = (pred_class_logits_base.argmax(dim=-1) == self.food_class_id).split(num_preds_per_image, dim=0)
        
        num_preds_per_image = torch.tensor(num_preds_per_image).long()
        
        # NOTE HDA here we filter the proposals for animal, food, and background
        # TODO: replace the hardcode of animal, food id to contiguous id
        for i, (
            pred_instance_fg, fg_inds_local, proposal, num_preds, bg_score_max_bool, animal_score_max_bool, food_score_max_bool
            ) in enumerate(
            zip(pred_instances_fg, fg_inds_local_list, proposals, num_preds_per_image, bg_score_max_bool_list, animal_score_max_bool_list, food_score_max_bool_list)
            ):
            
            pred_instance_fg_fixed_cats_bool = torch.logical_and(pred_instance_fg.pred_classes!=self.animal_class_id, pred_instance_fg.pred_classes!=self.food_class_id)
            fg_fixed_cats_inds_local = fg_inds_local[pred_instance_fg_fixed_cats_bool]
            fg_animal_inds_local = torch.tensor([i for i in torch.arange(num_preds).to(self.device)[animal_score_max_bool] if i not in fg_fixed_cats_inds_local]).long().to(self.device)
            fg_food_inds_local = torch.tensor([i for i in torch.arange(num_preds).to(self.device)[food_score_max_bool] if i not in fg_fixed_cats_inds_local]).long().to(self.device)
            bg_inds_local = torch.tensor([i for i in torch.arange(num_preds).to(self.device)[bg_score_max_bool] if i not in fg_fixed_cats_inds_local]).long().to(self.device)
            if self.training:
                assert proposal.has('gt_classes')
                # NOTE here we assume that to filter out proposals that have max score of animal but do not have ground truth id of animal
                fg_animal_inds_local = torch.tensor([ind for ind in fg_animal_inds_local if proposal.gt_classes[ind] in self.idmaps['hier2_animal_class_ids_global']]).long().to(self.device)
                fg_food_inds_local = torch.tensor([ind for ind in fg_food_inds_local if proposal.gt_classes[ind] in self.idmaps['hier2_food_class_ids_global']]).long().to(self.device)

            pred_instance_fg_fixed_cats = self.instance_filter(pred_instance_fg, pred_instance_fg_fixed_cats_bool)
            pred_instances_fg_fixed_cats.append(pred_instance_fg_fixed_cats)
            
            fg_animal_inds = torch.cat((fg_animal_inds, torch.sum(num_preds_per_image[:i])+fg_animal_inds_local)).long()
            fg_food_inds = torch.cat((fg_food_inds, torch.sum(num_preds_per_image[:i])+fg_food_inds_local)).long()
            bg_inds = torch.cat((bg_inds, torch.sum(num_preds_per_image[:i])+bg_inds_local)).long()
            
            proposals_animal.append(self.proposal_filter(proposal, fg_animal_inds_local, super_cat = 'animal'))
            proposals_food.append(self.proposal_filter(proposal, fg_food_inds_local, super_cat = 'food'))
            proposals_bg.append(self.proposal_filter(proposal, bg_inds_local, super_cat = 'bg'))

        return pred_instances_fg_fixed_cats, fg_animal_inds, fg_food_inds, bg_inds, proposals_animal, proposals_food, proposals_bg

    def instance_filter(self, instance, inds):
        filtered_instance = Instances(instance.image_size)
        filtered_instance.pred_boxes = Boxes(instance.pred_boxes.tensor[inds])
        filtered_instance.scores = instance.scores[inds]
        filtered_instance.pred_classes = instance.pred_classes[inds]
        if instance.has("gt_classes"):
            filtered_instance.gt_classes = instance.gt_classes[inds]
        return filtered_instance
        
    def proposal_filter(self, proposal, inds, super_cat: str):
        assert super_cat in ['bg', 'animal', 'food']

        proposal_filtered = Instances(proposal.image_size)
        proposal_filtered.proposal_boxes = Boxes(proposal.proposal_boxes.tensor[inds])
        proposal_filtered.objectness_logits = proposal.objectness_logits[inds]
        
        # only for training we have the gt_classes
        if proposal.has('gt_boxes'):
            proposal_filtered.gt_boxes = Boxes(proposal.gt_boxes.tensor[inds])

            assert proposal.has('gt_classes')

            if self.with_gt == True:
                proposal_filtered.gt_classes = proposal.gt_classes[inds]
            else:

                idmap_global_reversed = self.idmaps['idmap_hier2_global_reversed']
                idmap_local = self.idmaps['idmap_hier2_{}'.format(super_cat)]
                
                # idmap_local map the child cat id to contiguous local id
                # NOTE HDA the num_classes_base here should be 60 child classes rather than 42
                gt_classes_filtered = proposal.gt_classes[inds]
                gt_classes_mapped = torch.zeros(gt_classes_filtered.shape).to(self.device)

                if super_cat == 'bg':
                    bg_class_id_global = self.num_classes_dict['hier2_fg']+self.num_classes_dict['hier2_bg']
                    bg_class_id_local = self.num_classes_dict['hier2_bg']
                    other_class_ids_global = self.idmaps['hier2_fg_class_ids_global'] + [bg_class_id_global]
                    for i, gt_class in enumerate(gt_classes_filtered.cpu().numpy()):
                        gt_classes_mapped[i] = bg_class_id_local if gt_class in other_class_ids_global else idmap_local[idmap_global_reversed[gt_class]]
                else:
                    for i, gt_class in enumerate(gt_classes_filtered.cpu().numpy()):
                        gt_classes_mapped[i] = idmap_local[idmap_global_reversed[gt_class]]

                proposal_filtered.gt_classes = gt_classes_mapped.long()

        return proposal_filtered

    # TODO
    # check the output
    # check the output of the proposal
    def extract_features(self, images, features, proposals, targets=None, extract_gt_box_features = False):
        """
        Extract features from the input data
        """
        del images
        assert self.training, "Model was changed to eval mode!"
        proposals = self.label_and_sample_proposals(proposals, targets)
        # del targets

        features_list = [features[f] for f in self.in_features]
        extracted_info = self._extract_features_box(features_list, proposals)
        del targets
        return extracted_info
        
    def _extract_features_box(self, features, proposals):
        """
        Forward logic of the box prediction branch for extracting features.

        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, extracted features (list[Tensor]).
        """
        # TODO replace proposals with targets and x.gt_boxes, one x corresponds to one img, also store gt.classes
        assert self.training, "Model was changed to eval mode!"
        box_features = self.box_pooler(
            features, [x.proposal_boxes for x in proposals]
        )
        box_features = self.box_head(box_features)

        # NOTE from the pred_class_logits we can get the pred class from base detetor for each proposal
        pred_class_logits_base, pred_proposal_deltas_base = self.box_predictor_base(box_features)
        # fg_inds, bg_inds, proposals_base, proposals_novel = self.detection_filter(proposals, pred_class_logits_base, pred_proposal_deltas_base)
        _, fg_animal_inds, fg_food_inds, bg_inds, proposals_animal, proposals_food, proposals_bg = self.detection_filter(
            proposals, pred_class_logits_base, pred_proposal_deltas_base
        )
        box_features_animal = box_features[fg_animal_inds] if fg_animal_inds.shape[0] != 0 else torch.tensor([]).to(self.device)
        box_features_food = box_features[fg_food_inds] if fg_food_inds.shape[0] != 0 else torch.tensor([]).to(self.device)
        box_features_bg = box_features[bg_inds] if bg_inds.shape[0] != 0 else torch.tensor([]).to(self.device)

        # NOTE experiment for rebuttal
        pred_classes = pred_class_logits_base.argmax(dim=1)
        true_classes = torch.cat(tuple(proposal.gt_classes for proposal in proposals))
        extracted_info = {
            'proposals_animal': proposals_animal,
            'box_features_animal': box_features_animal,
            'proposals_food': proposals_food,
            'box_features_food': box_features_food,
            'proposals_bg': proposals_bg,
            'box_features_bg': box_features_bg,
            'pred': pred_classes,
            'true': true_classes,
            }

        # print('extracted_info')
        # embed()
        return extracted_info

    def losses_from_features(self, box_features, proposals, weights = None, super_cat:str = 'bg'):
        """
        Forward logic of the box prediction branch for computing losses.

        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
        """
        assert self.training, "Model was changed to eval mode!"
        # TODO for training, the following code only has to be done once, as both the box_features and box_predictor_base are fixed
        # pred_class_logits_base, pred_proposal_deltas_base = self.box_predictor_base(box_features)
        # fg_inds, bg_inds, proposals_base, proposals_novel = self.detection_filter(proposals, pred_class_logits_base, pred_proposal_deltas_base)
        # pred_class_logits_base, pred_proposal_deltas_base = pred_class_logits_base[fg_inds], pred_proposal_deltas_base[fg_inds]
       
        assert super_cat in ['bg', 'animal', 'food']
        if super_cat == 'bg':
            pred_class_logits_bg, pred_proposal_deltas_bg = self.box_predictor_bg(
                box_features, weights
            )
            del box_features
            
            # output loss for a certain super class in training, same for forward of SGD
            outputs_bg = FastRCNNOutputs(
                self.box2box_transform,
                pred_class_logits_bg,
                pred_proposal_deltas_bg,
                proposals,
                self.smooth_l1_beta,
            )
                
            return outputs_bg.losses()

        elif super_cat =='animal':
            pred_class_logits_animal, pred_proposal_deltas_animal = self.box_predictor_animal(
                box_features, weights
            )
            del box_features

            outputs_animal = FastRCNNOutputs(
                self.box2box_transform,
                pred_class_logits_animal,
                pred_proposal_deltas_animal,
                proposals,
                self.smooth_l1_beta,
                super_base_class=True,
            )

            # self.iter_count +=1
            # if self.iter_count % 10 == 0:
            #     print('animal')
            #     embed()

            return outputs_animal.losses()
            # return {'loss_cls_{}'.format(super_cat): self.cls_loss(pred_class_logits_animal, proposals)}
            
        elif super_cat == 'food':
            pred_class_logits_food, pred_proposal_deltas_food = self.box_predictor_food(
                box_features, weights
            )
            del box_features

            outputs_food = FastRCNNOutputs(
                self.box2box_transform,
                pred_class_logits_food,
                pred_proposal_deltas_food,
                proposals,
                self.smooth_l1_beta,
                super_base_class=True,
            )

            return outputs_food.losses()
            # return {'loss_cls_{}'.format(super_cat): self.cls_loss(pred_class_logits_food, proposals)}
            
    
    def cls_loss(self, pred_class_logits, proposals):
        assert proposals[0].has("gt_classes")
        gt_classes = cat([p.gt_classes for p in proposals], dim=0)
        
        return F.cross_entropy(
            pred_class_logits, gt_classes, reduction="mean"
        )

@ROI_HEADS_REGISTRY.register()
class TwoStageROIHeads_lvis_rc(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    The cropped rois go to separate branches directly.
    This way, it is easier to make separate abstractions for different branches.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    def __init__(self, cfg, input_shape):
        super(TwoStageROIHeads_lvis_rc, self).__init__(cfg, input_shape)
        self._init_box_head(cfg)

    def _init_box_head(self, cfg):
        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [self.feature_channels[f] for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.box_head = build_box_head(
            cfg,
            ShapeSpec(
                channels=in_channels,
                height=pooler_resolution,
                width=pooler_resolution,
            ),
        )
        output_layer = cfg.MODEL.ROI_HEADS.OUTPUT_LAYER
        
        # NOTE for lvis rc implementation
        # init the model: base: f+c, novel: c+r
        # filter the fg_instance to have the detections of only f
        # proposal filtering, base: f, novel: c+r
        # note the id mapping, in proposal filtering and merge instance
        # proposal filtering, only filter the proposals for novel detector (bg score: c+bg), map of id, same as now
        # merge instance, map the id of fg instance back, the idmap of base needs to be changed.

        # TODO LVIS modify the following params in the config accordingly
        # NOTE in LVIS training, we trainn it directly on the Rare dataset, so num_classes is 454 when label the bg id
        self.lvis = True if 'lvis' in cfg.DATASETS.TRAIN[0] else False

        self.num_classes_base = cfg.MODEL.ROI_HEADS.NUM_CLASSES_BASE # f 315
        self.num_classes_novel = cfg.MODEL.ROI_HEADS.NUM_CLASSES_NOVEL # 915 = c 461 + r 454
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        metadata = self.get_metadata(cfg.DATASETS.TRAIN[0])
        # metadata = _get_builtin_metadata('coco_fewshot')
        # TODO lvis c, adjust the idmap here
        self.idmaps = self.init_idmaps(metadata)
        self.metadata = metadata

        self.box_predictor_base = ROI_HEADS_OUTPUT_REGISTRY.get(output_layer)(
            cfg,
            self.box_head.output_size,
            # NOTE HDA num of base classes should be 42 for the first hier level (also related to the weight initialization)
            self.num_classes_base+461,
            self.cls_agnostic_bbox_reg,
        )
        # train it using CG
        self.box_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(output_layer)(
            cfg,
            self.box_head.output_size,
            self.num_classes_novel,
            self.cls_agnostic_bbox_reg,
        )
        # NOTE HDA add two more roi heads for the child classes of animal and food
        self.with_gt = False

    def get_metadata(self, dataset_name):
        metadata = dict({})
        # NOTE HDA the metadata for coco to get is coco_fewshot_base_all
        # NOTE LVIS add the meta data of lvis needed for proposal filtering
        if 'coco' in dataset_name:
            metadata = _get_builtin_metadata('coco_fewshot')
        elif 'voc' in dataset_name:
            sid = int(dataset_name.split('all')[1].split('_')[0])
            init_metadata = _get_builtin_metadata('pascal_voc_fewshot')
            metadata['thing_classes'] = init_metadata['thing_classes'][sid]
            metadata['base_classes'] = init_metadata['base_classes'][sid]
            metadata['novel_classes'] = init_metadata['novel_classes'][sid]
            metadata['thing_dataset_id_to_contiguous_id'] = {i:i for i,v in enumerate(metadata['thing_classes'])}
            metadata['base_dataset_id_to_contiguous_id'] = {i:i for i,v in enumerate(metadata['base_classes'])}
            metadata['novel_dataset_id_to_contiguous_id'] = {len(metadata['base_classes'])+i:i for i,v in enumerate(metadata['novel_classes'])}
        elif 'lvis' in dataset_name:
            # NOTE the metadata used here should contain thing, base, novel, which coulld be different to the metadata of the dataset
            # if the novel detector is trained directly on the Rare dataset, then the metadata here will only be used in inference.
            assert len(LVIS_CATEGORIES) == 1230 
            cat_ids = [k["id"] for k in LVIS_CATEGORIES]
            assert min(cat_ids) == 1 and max(cat_ids) == len(cat_ids), "Category ids are not in [1, #categories], as expected"
            
            # Ensure that the category list is sorted by id
            lvis_categories = [k for k in sorted(LVIS_CATEGORIES, key=lambda x: x["id"])]
            thing_classes = [k["synonyms"][0] for k in lvis_categories]
            
            assert self.num_classes_novel == 915
            lvis_categories_novel = [k for k in sorted(LVIS_CATEGORIES, key=lambda x: x["id"]) if k["frequency"] in ['c', 'r']]
            novel_classes = [k["synonyms"][0] for k in lvis_categories_novel]
            
            lvis_categories_base = [k for k in sorted(LVIS_CATEGORIES, key=lambda x: x["id"]) if k["frequency"] in ['f', 'c']]
            base_classes = [k["synonyms"][0] for k in lvis_categories_base if k["frequency"]=='f']


            metadata["thing_classes"]= thing_classes
            metadata["thing_dataset_id_to_contiguous_id"] = {x["id"]:i for i,x in enumerate(lvis_categories)}
            metadata["novel_classes"] = novel_classes
            metadata["novel_dataset_id_to_contiguous_id"] = {x["id"]:i for i,x in enumerate(lvis_categories_novel)}
            # NOTE the base map should have both f and c and then filter out c, to be consistent with the base detector
            metadata["base_classes"] = base_classes
            metadata["base_dataset_id_to_contiguous_id"] = {x["id"]:i for i,x in enumerate(lvis_categories_base) if x["frequency"]=='f'}
            
            metadata["base_detector_bg_ids"] = [i for i, x in enumerate(lvis_categories_base) if x["frequency"]=='c']+[len(lvis_categories_base)]
            metadata["novel_detector_c_ids"] = [i for i, x in enumerate(lvis_categories_novel) if x["frequency"]=='c']
            metadata["novel_detector_r_ids"] = [i for i, x in enumerate(lvis_categories_novel) if x["frequency"]=='r']

        return metadata

    # TODO LVIS new init_idmaps for LVIS or put it in a separate class
    def init_idmaps(self, metadata):
        idmap_global = metadata['thing_dataset_id_to_contiguous_id']
        idmap_global_reversed = {v: k for k, v in idmap_global.items()}

        idmap_base = metadata['base_dataset_id_to_contiguous_id']
        idmap_base_reversed = {v: k for k, v in idmap_base.items()}
        base_class_ids_global = [idmap_global[k] for k in idmap_base.keys()]

        idmap_novel = metadata['novel_dataset_id_to_contiguous_id']
        idmap_novel_reversed = {v: k for k, v in idmap_novel.items()}
        novel_class_ids_global = [idmap_global[k] for k in idmap_novel.keys()]

        return {
            'idmap_global': idmap_global,
            'idmap_global_reversed': idmap_global_reversed,
            'idmap_base': idmap_base,
            'idmap_base_reversed': idmap_base_reversed,
            'base_class_ids_global': base_class_ids_global,
            'idmap_novel': idmap_novel,
            'idmap_novel_reversed': idmap_novel_reversed,
            'novel_class_ids_global': novel_class_ids_global,
        }
        
    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        # NOTE gt
        elif targets is not None:
            proposals = self.label_and_sample_proposals(proposals, targets)
            self.with_gt = True
        
        del targets

        features_list = [features[f] for f in self.in_features]
        
        if self.training:
            losses = self._forward_box(features_list, proposals)
            return proposals, losses
        else:
            pred_instances = self._forward_box(features_list, proposals)
            return pred_instances, {}
    
    def _forward_box(self, features, proposals):
        """
        Forward logic of the box prediction branch.

        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        box_features = self.box_pooler(
            features, [x.proposal_boxes for x in proposals]
        )
        box_features = self.box_head(box_features)

        # NOTE If the dataset is LVIS and it is in training mode, then filtering is not needed as train the model directly on Rare dataset
        # Overall the forward_box function is only used for batch case, for both training and inference
        # actually currently this function only support the training for LVIS dataset, in SCG case (also SGD in the future)
        
        # if self.training and self.lvis:
        #     pred_class_logits_novel, pred_proposal_deltas_novel = self.box_predictor(box_features)
        #     proposals_novel = proposals
        # else:
        pred_class_logits_base, pred_proposal_deltas_base = self.box_predictor_base(box_features)
        pred_instances_base, bg_inds, proposals_novel = self.detection_filter(proposals, pred_class_logits_base, pred_proposal_deltas_base)
        
        # selected base proposals, and corresponding logits and deltas 
        # pred_class_logits_base, pred_proposal_deltas_base = pred_class_logits_base[fg_inds], pred_proposal_deltas_base[fg_inds]
        pred_class_logits_novel, pred_proposal_deltas_novel = self.box_predictor(box_features[bg_inds])
        
        del box_features

        # NOTE gt
        outputs_novel = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits_novel,
            pred_proposal_deltas_novel,
            proposals_novel,
            self.smooth_l1_beta,
        )

        if self.training:
            return outputs_novel.losses()
        else:
            pred_instances_novel, _ = outputs_novel.inference(
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_detections_per_img,
                with_gt = self.with_gt,
            )
            pred_instances = self.merge_instances(pred_instances_base, pred_instances_novel)
        
            return pred_instances
  
    def merge_instances(self, pred_instances_base, pred_instances_novel):
        idmap_global = self.idmaps['idmap_global']
        idmap_base_reversed = self.idmaps['idmap_base_reversed']
        idmap_novel_reversed = self.idmaps['idmap_novel_reversed']

        pred_instances = list([])
        for instance_base, instance_novel in zip(pred_instances_base, pred_instances_novel):
            assert instance_base.image_size == instance_novel.image_size
            instance = Instances(instance_base.image_size)
            instance.pred_boxes = Boxes(torch.cat((instance_base.pred_boxes.tensor, instance_novel.pred_boxes.tensor)))
            instance.scores = torch.cat((instance_base.scores, instance_novel.scores))

            pred_classes_mapped_base = torch.zeros(instance_base.pred_classes.shape).to(self.device)
            for i, pred_class in enumerate(instance_base.pred_classes.cpu().numpy()):
                pred_classes_mapped_base[i] = idmap_global[idmap_base_reversed[pred_class]]

            pred_classes_mapped_novel = torch.zeros(instance_novel.pred_classes.shape).to(self.device)
            for i, pred_class in enumerate(instance_novel.pred_classes.cpu().numpy()):
                pred_classes_mapped_novel[i] = idmap_global[idmap_novel_reversed[pred_class]]
            
            instance.pred_classes = torch.cat((pred_classes_mapped_base, pred_classes_mapped_novel))
            # NOTE gt
            if instance_base.has("gt_classes"):
                instance.gt_classes = torch.cat((instance_base.gt_classes, instance_novel.gt_classes))
            pred_instances.append(instance)
    
        return pred_instances

    def detection_filter(self, proposals, pred_class_logits_base, pred_proposal_deltas_base):
        # NOTE for test
        outputs_base_simulator = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits_base,
            pred_proposal_deltas_base,
            proposals,
            self.smooth_l1_beta,
        )

        # NOTE gt
        pred_instances_fg, fg_inds_local_list = outputs_base_simulator.inference(
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_detections_per_img,
            with_gt = self.with_gt,
        )
        pred_instances_fg_f = list([])
        # embed()
        # NOTE lvis rc, local base ids for f and c respectively, to filter the fg instance and select background proposals
        num_preds_per_image = [len(p) for p in proposals]
        # fg_inds = torch.tensor([]).to(self.device)
        bg_inds = torch.tensor([]).to(self.device)
        # NOTE HDA proposals_base will also be needed in training
        # proposals_base = None if self.training else list([]) 
        proposals_novel = list([])
        
        pred_classes = pred_class_logits_base.argmax(dim=1)
        # bg_class_id = pred_class_logits_base.shape[1]-1
        # bg_score_max_bool_list = (pred_classes == bg_class_id).split(num_preds_per_image, dim=0)
        bg_class_ids = set(self.metadata["base_detector_bg_ids"])
        bg_score_max_bool_list = torch.tensor([pred_class in bg_class_ids for pred_class in pred_classes.cpu().numpy()]).to(self.device).split(num_preds_per_image, dim=0)
        
        num_preds_per_image = torch.tensor(num_preds_per_image).long()
    
        # NOTE HDA here we filter the proposals for animal, food, and background
        for i, (pred_instance_fg, fg_inds_local, proposal, num_preds, bg_score_max_bool) in enumerate(zip(pred_instances_fg, fg_inds_local_list, proposals, num_preds_per_image, bg_score_max_bool_list)):
            fg_inds_local_f_bool = torch.tensor([pred_class not in bg_class_ids for pred_class in pred_instance_fg.pred_classes.cpu().numpy()]).to(self.device)
            fg_inds_local_f = fg_inds_local[fg_inds_local_f_bool]
            bg_inds_local = torch.tensor([i for i in torch.arange(num_preds).to(self.device)[bg_score_max_bool] if i not in fg_inds_local_f]).long().to(self.device)
        
            # NOTE TODO check here the index out of bound in instance filter?
            pred_instance_fg_f = self.instance_filter(pred_instance_fg, fg_inds_local_f_bool)
            pred_instances_fg_f.append(pred_instance_fg_f)

            bg_inds = torch.cat((bg_inds, torch.sum(num_preds_per_image[:i])+bg_inds_local)).long()
            proposals_novel.append(self.proposal_filter(proposal, bg_inds_local, novel = True))
            
        # return pred_instances_fg, fg_inds, bg_inds, proposals_base, proposals_novel
        return pred_instances_fg_f, bg_inds, proposals_novel

    def instance_filter(self, instance, inds):
        filtered_instance = Instances(instance.image_size)
        filtered_instance.pred_boxes = Boxes(instance.pred_boxes.tensor[inds])
        filtered_instance.scores = instance.scores[inds]
        filtered_instance.pred_classes = instance.pred_classes[inds]
        if instance.has("gt_classes"):
            filtered_instance.gt_classes = instance.gt_classes[inds]
        return filtered_instance

    def proposal_filter(self, proposal, inds, novel = False):
        proposal_filtered = Instances(proposal.image_size)
        proposal_filtered.proposal_boxes = Boxes(proposal.proposal_boxes.tensor[inds])
        proposal_filtered.objectness_logits = proposal.objectness_logits[inds]
        
        # only for training we have the gt_classes
        if proposal.has('gt_boxes'):
            proposal_filtered.gt_boxes = Boxes(proposal.gt_boxes.tensor[inds])

            assert proposal.has('gt_classes')
            if self.with_gt == True:
                # If with_gt is True, it means that it is in test phase, rather than training phase
                proposal_filtered.gt_classes = proposal.gt_classes[inds]
            else:
                # NOTE add part for gt in test
                idmap_global = self.idmaps['idmap_global']
                idmap_global_reversed = self.idmaps['idmap_global_reversed']
                idmap_local = self.idmaps['idmap_novel'] if novel is True else self.idmaps['idmap_base']
                # NOTE HDA the num_classes_base here should be 60 child classes rather than 42
                bg_class_id_global = self.num_classes_base+self.num_classes_novel
                bg_class_id_local = self.num_classes_novel if novel is True else self.num_classes_base
                other_class_ids_global = self.idmaps['base_class_ids_global'] if novel is True else self.idmaps['novel_class_ids_global']
                other_class_ids_global = list(other_class_ids_global) + [bg_class_id_global]

                gt_classes_filtered = proposal.gt_classes[inds]
                gt_classes_mapped = torch.zeros(gt_classes_filtered.shape).to(self.device)
                for i, gt_class in enumerate(gt_classes_filtered.cpu().numpy()):
                    gt_classes_mapped[i] = bg_class_id_local if gt_class in other_class_ids_global else idmap_local[idmap_global_reversed[gt_class]]
            
                proposal_filtered.gt_classes = gt_classes_mapped.long()

        return proposal_filtered

    # TODO LVIS compile this function with batch SCG
    def extract_features(self, images, features, proposals, targets=None, extract_gt_box_features = False):
        """
        Extract features from the input data
        """
        del images
        assert self.training, "Model was changed to eval mode!"
        proposals = self.label_and_sample_proposals(proposals, targets)
        # del targets

        features_list = [features[f] for f in self.in_features]
        proposals_novel, box_features_novel = self._extract_features_box(features_list, proposals)
        if not extract_gt_box_features:
            del targets
            return proposals_novel, box_features_novel
        else:
            gt_box_features, gt_classes = self._extract_gt_features_box(features_list, targets)
            del targets
            return proposals_novel, box_features_novel, gt_box_features, gt_classes

    def _extract_features_box(self, features, proposals):
        """
        Forward logic of the box prediction branch for extracting features.

        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, extracted features (list[Tensor]).
        """
        # TODO replace proposals with targets and x.gt_boxes, one x corresponds to one img, also store gt.classes
        assert self.training, "Model was changed to eval mode!"
        box_features = self.box_pooler(
            features, [x.proposal_boxes for x in proposals]
        )
        box_features = self.box_head(box_features)
        
        # if self.lvis:
        #     # NOTE for LVIS, the model will be trained directly on novel dataset
        #     proposals_novel = proposals
        #     box_features_novel = box_features
        # else:
        pred_class_logits_base, pred_proposal_deltas_base = self.box_predictor_base(box_features)
        pred_instances_base, bg_inds, proposals_novel = self.detection_filter(proposals, pred_class_logits_base, pred_proposal_deltas_base)

        box_features_novel = box_features[bg_inds] if bg_inds.shape[0] != 0 else torch.tensor([]).to(self.device)

        return proposals_novel, box_features_novel

    # TODO filter the novel classes and only extract the novel classes
    def _extract_gt_features_box(self, features, targets):
        """
        Forward logic of the box prediction branch for extracting features for ground truth bbox.

        Args:
            features (list[Tensor]): #level input features for box prediction
            targets (list[Instances]): the per-image ground truth of Instances.
                Each has fields "gt_classes", "gt_boxes".

        Returns:
            In training, extracted features (list[Tensor]), gt_classes (list[Tensor])
        """
        # TODO this has not been adapted for the two stage training
        assert self.training, "Model was changed to eval mode!"                
        # extract features for all ground truth boxes
        gt_box_features = self.box_pooler(
            features, [x.gt_boxes for x in targets]
        )
        gt_box_features = self.box_head(gt_box_features)
        
        gt_classes = [x.gt_classes for x in targets]
        
        return gt_box_features, gt_classes

    def losses_from_features(self, box_features, proposals, weights = None, super_cat:str = None):
        """
        Forward logic of the box prediction branch for computing losses.

        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
        """
        assert self.training, "Model was changed to eval mode!"
       
        pred_class_logits_novel, pred_proposal_deltas_novel = self.box_predictor(
            box_features, weights
        )
        del box_features
        # print('loss')
        # embed()
        outputs_novel = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits_novel,
            pred_proposal_deltas_novel,
            proposals,
            self.smooth_l1_beta,
        )

        return outputs_novel.losses()

