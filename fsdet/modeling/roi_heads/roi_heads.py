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
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

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

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
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

    def losses_from_features(self, box_features, proposals, weights = None):
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

        # TODO change the number of classes to cfg params
        # TODO add pascal voc 15, 5
        self.num_classes_base = cfg.MODEL.ROI_HEADS.NUM_CLASSES_BASE
        self.num_classes_novel = cfg.MODEL.ROI_HEADS.NUM_CLASSES_NOVEL
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # TODO currently, the metadata has not been adapted for meta learning
        # for voc, the idmap of metadata needs to be set up here, novel is the last five
        metadata = self.get_metadata(cfg.DATASETS.TRAIN[0])
        # metadata = _get_builtin_metadata('coco_fewshot')
        self.idmaps = self.init_idmaps(metadata)
        self.metadata = metadata

        # TODO load the pretrained weights and freeze it 
        self.box_predictor_base = ROI_HEADS_OUTPUT_REGISTRY.get(output_layer)(
            cfg,
            self.box_head.output_size,
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

    def get_metadata(self, dataset_name):
        metadata = dict({})
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
        return metadata

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

        # NOTE box predictor should be frozen
        pred_class_logits_base, pred_proposal_deltas_base = self.box_predictor_base(box_features)
        fg_inds, bg_inds, proposals_base, proposals_novel = self.detection_filter(proposals, pred_class_logits_base, pred_proposal_deltas_base)
        
        pred_class_logits_base, pred_proposal_deltas_base = pred_class_logits_base[fg_inds], pred_proposal_deltas_base[fg_inds]
        pred_class_logits_novel, pred_proposal_deltas_novel = self.box_predictor(box_features[bg_inds])
        del box_features

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
            outputs_base = FastRCNNOutputs(
                self.box2box_transform,
                pred_class_logits_base,
                pred_proposal_deltas_base,
                proposals_base,
                self.smooth_l1_beta,
            )
            pred_instances_base, _ = outputs_base.inference(
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_detections_per_img,
            )
            pred_instances_novel, _ = outputs_novel.inference(
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_detections_per_img,
            )
            # print('infer done')
            # embed()

            # idmap_global = self.metadata['thing_dataset_id_to_contiguous_id']
            # idmap_base_reversed = {v: k for k, v in self.metadata['base_dataset_id_to_contiguous_id'].items()}
            # idmap_novel_reversed = {v: k for k, v in self.metadata['novel_dataset_id_to_contiguous_id'].items()}

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

                pred_instances.append(instance)
            # print('result cat done')
            # embed()
            return pred_instances
    

    def detection_filter(self, proposals, pred_class_logits_base, pred_proposal_deltas_base):
        # num_preds_per_image = [len(p) for p in proposals]
        # score_thresh = 0.05 # change it to cfg later
        # scores = F.softmax(pred_class_logits_base, dim=-1)
        # filter_mask = scores[:,:-1]>score_thresh
        # filter_inds = filter_mask.nonzero()
        # fg_inds = filter_inds[:,0].unique()
        # bg_inds = torch.tensor([i for i in torch.arange(scores.shape[0]).to(self.device) if i not in fg_inds])

        # scores_local_list = scores.split(num_preds_per_image, dim=0)
        # proposals_base = None if self.training else list([]) 
        # proposals_novel = list([])
        # for scores_local, proposal in zip(scores_local_list, proposals):
        #     filter_mask_local = scores_local[:,:-1]>score_thresh
        #     filter_inds_local = filter_mask_local.nonzero()
        #     fg_inds_local = filter_inds_local[:,0].unique()
        #     bg_inds_local = torch.tensor([i for i in torch.arange(scores_local.shape[0]).to(self.device) if i not in fg_inds_local])
            

        #     if proposals_base is not None:
        #         proposals_base.append(self.proposal_filter(proposal, fg_inds_local, novel = False))
        #     proposals_novel.append(self.proposal_filter(proposal, bg_inds_local, novel = True))
        
        # NOTE for test
        outputs_base_simulator = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits_base,
            pred_proposal_deltas_base,
            proposals,
            self.smooth_l1_beta,
        )

        _, fg_inds_local_list = outputs_base_simulator.inference(
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_detections_per_img,
        )
        # embed()

        num_preds_per_image = [len(p) for p in proposals]
        fg_inds = torch.tensor([]).to(self.device)
        bg_inds = torch.tensor([]).to(self.device)
        proposals_base = None if self.training else list([]) 
        proposals_novel = list([])
        
        pred_classes = pred_class_logits_base.argmax(dim=1)
        bg_class_id = pred_class_logits_base.shape[1]-1
        bg_score_max_bool_list = (pred_classes == bg_class_id).split(num_preds_per_image, dim=0)
        
        num_preds_per_image = torch.tensor(num_preds_per_image).long()
        # TODO modify the filtering procedure here for meta learning of novel predictor, to train and test it on novel dataset only
        # if self.training:
        #     for i, (proposal, num_preds) in enumerate(zip(proposals, num_preds_per_image)):
        #         if proposal.gt_classes[0] in self.idmaps['base_class_ids_global']:
        #             continue
        #         else:
        #             bg_inds_local = torch.tensor([i for i in torch.arange(num_preds)]).long().to(self.device)
        #             bg_inds = torch.cat((bg_inds, torch.sum(num_preds_per_image[:i])+bg_inds_local)).long()
        #             proposals_novel.append(self.proposal_filter(proposal, bg_inds_local, novel = True))
                
        #         # print('iter')
        #         # embed()
    
        # else:
        for i, (fg_inds_local, proposal, num_preds, bg_score_max_bool) in enumerate(zip(fg_inds_local_list, proposals, num_preds_per_image, bg_score_max_bool_list)):

            fg_inds_local, _ = fg_inds_local.long().unique().sort()
            bg_inds_local = torch.tensor([i for i in torch.arange(num_preds).to(self.device)[bg_score_max_bool] if i not in fg_inds_local]).long().to(self.device)
            
            fg_inds = torch.cat((fg_inds, torch.sum(num_preds_per_image[:i])+fg_inds_local)).long()
            bg_inds = torch.cat((bg_inds, torch.sum(num_preds_per_image[:i])+bg_inds_local)).long()

            if proposals_base is not None:
                proposals_base.append(self.proposal_filter(proposal, fg_inds_local, novel = False))

            proposals_novel.append(self.proposal_filter(proposal, bg_inds_local, novel = True))
            
            # print('iter')
            # embed()

        # pred_classes = pred_class_logits_base.argmax(dim=1)
        # bg_class_id = pred_class_logits_base.shape[1]-1
        # fg_inds = torch.arange(len(pred_classes))[pred_classes!=bg_class_id].long().to(self.device)
        # bg_inds = torch.arange(len(pred_classes))[pred_classes==bg_class_id].long().to(self.device)
        # pred_class_logits_base_list = pred_class_logits_base.split(num_preds_per_image, dim=0)
        # for i, (proposal, num_preds, pred_class_logits_base_local) in enumerate(zip(proposals, num_preds_per_image, pred_class_logits_base_list)):
        #     pred_classes_local = pred_class_logits_base_local.argmax(dim=1)
        #     fg_inds_local = torch.arange(len(pred_classes_local))[pred_classes_local!=bg_class_id].long().to(self.device)
        #     bg_inds_local = torch.arange(len(pred_classes_local))[pred_classes_local==bg_class_id].long().to(self.device)

        #     if proposals_base is not None:
        #         proposals_base.append(self.proposal_filter(proposal, fg_inds_local, novel = False))
        #     proposals_novel.append(self.proposal_filter(proposal, bg_inds_local, novel = True))
        
        # print('filter done')
        # embed()
        
        return fg_inds, bg_inds, proposals_base, proposals_novel

    def proposal_filter(self, proposal, inds, novel = False):
        proposal_filtered = Instances(proposal.image_size)
        proposal_filtered.proposal_boxes = Boxes(proposal.proposal_boxes.tensor[inds])
        proposal_filtered.objectness_logits = proposal.objectness_logits[inds]

        if proposal.has('gt_boxes'):
            proposal_filtered.gt_boxes = Boxes(proposal.gt_boxes.tensor[inds])

            assert proposal.has('gt_classes')
            
            idmap_global = self.idmaps['idmap_global']
            idmap_global_reversed = self.idmaps['idmap_global_reversed']
            idmap_local = self.idmaps['idmap_novel'] if novel is True else self.idmaps['idmap_base']
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


    def merge_predictions(self, fg_inds, bg_inds, pred_class_logits_base, pred_proposal_deltas_base, pred_class_logits_novel, pred_proposal_deltas_novel):
        """"Merge the classification score and bbox deltas from both base and novel predictors"""

        pred_class_logits = torch.zeros(len(fg_inds)+len(bg_inds), self.num_classes+1).to(self.device)
        pred_class_logits_base_part = (torch.ones(len(fg_inds), self.num_classes+1)*-10000).to(self.device)
        pred_class_logits_novel_part = (torch.ones(len(bg_inds), self.num_classes+1)*-10000).to(self.device)
        
        pred_proposal_deltas = torch.zeros(len(fg_inds)+len(bg_inds), self.num_classes*4).to(self.device)
        pred_proposal_deltas_base_part = torch.zeros(len(fg_inds), self.num_classes*4).to(self.device)
        pred_proposal_deltas_novel_part = torch.zeros(len(bg_inds), self.num_classes*4).to(self.device)

        IDMAP = self.metadata['thing_dataset_id_to_contiguous_id']

        for k, i in self.metadata['base_dataset_id_to_contiguous_id'].items():
            pred_class_logits_base_part[:, IDMAP[k]] = pred_class_logits_base[fg_inds,i]
            pred_proposal_deltas_base_part[:, IDMAP[k]*4:(IDMAP[k]+1)*4] = pred_proposal_deltas_base[fg_inds, i*4:(i+1)*4]
        # TODO for test
        pred_class_logits_base_part[:,-1] = pred_class_logits_base[fg_inds, -1]

        for k, i in self.metadata['novel_dataset_id_to_contiguous_id'].items():
            pred_class_logits_novel_part[:, IDMAP[k]] = pred_class_logits_novel[:,i]
            pred_proposal_deltas_novel_part[:, IDMAP[k]*4:(IDMAP[k]+1)*4] = pred_proposal_deltas_novel[:, i*4:(i+1)*4]
        # # assign the score for background
        # pred_class_logits_novel_part[:,-1] = pred_class_logits_novel[:,-1]

        for k, i in self.metadata['base_dataset_id_to_contiguous_id'].items():
            pred_class_logits_novel_part[:, IDMAP[k]] = pred_class_logits_base[bg_inds,i]
            pred_proposal_deltas_novel_part[:, IDMAP[k]*4:(IDMAP[k]+1)*4] = pred_proposal_deltas_base[bg_inds, i*4:(i+1)*4]
        # TODO for test
        pred_class_logits_novel_part[:,-1] = pred_class_logits_base[bg_inds, -1]

        pred_class_logits[fg_inds, :] = pred_class_logits_base_part
        pred_class_logits[bg_inds, :] = pred_class_logits_novel_part

        pred_proposal_deltas[fg_inds, :] = pred_proposal_deltas_base_part
        pred_proposal_deltas[bg_inds, :] = pred_proposal_deltas_novel_part

        return pred_class_logits, pred_proposal_deltas

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

        pred_class_logits_base, pred_proposal_deltas_base = self.box_predictor_base(box_features)
        fg_inds, bg_inds, proposals_base, proposals_novel = self.detection_filter(proposals, pred_class_logits_base, pred_proposal_deltas_base)
    
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

    def losses_from_features(self, box_features, proposals, weights = None):
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
        # pred_class_logits_base, pred_proposal_deltas_base = self.box_predictor_base(
        #     box_features
        # )
        
        # # Get the box features of background
        # pred_classes = pred_class_logits_base.argmax(dim=1)
        # bg_class_id = pred_class_logits_base.shape[1]-1
        # fg_inds = torch.arange(len(pred_classes))[pred_classes!=bg_class_id]
        # bg_inds = torch.arange(len(pred_classes))[pred_classes==bg_class_id]
        
        # pred_class_logits_novel, pred_proposal_deltas_novel = self.box_predictor_novel(
        #     box_features[bg_inds], weights
        # )
        # del box_features
        # pred_class_logits, pred_proposal_deltas = self.merge_predictions(fg_inds, bg_inds, pred_class_logits_base, pred_proposal_deltas_base, 
        #                                                                  pred_class_logits_novel, pred_proposal_deltas_novel)
        
        # outputs = FastRCNNOutputs(
        #     self.box2box_transform,
        #     pred_class_logits,
        #     pred_proposal_deltas,
        #     proposals,
        #     self.smooth_l1_beta,
        # )
        # # print('logits')
        # # embed()
        # return outputs.losses()
        
        # TODO for training, the following code only has to be done once, as both the box_features and box_predictor_base are fixed
        # pred_class_logits_base, pred_proposal_deltas_base = self.box_predictor_base(box_features)
        # fg_inds, bg_inds, proposals_base, proposals_novel = self.detection_filter(proposals, pred_class_logits_base, pred_proposal_deltas_base)
        # pred_class_logits_base, pred_proposal_deltas_base = pred_class_logits_base[fg_inds], pred_proposal_deltas_base[fg_inds]
       
        pred_class_logits_novel, pred_proposal_deltas_novel = self.box_predictor(
            box_features, weights
        )
        del box_features
        
        outputs_novel = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits_novel,
            pred_proposal_deltas_novel,
            proposals,
            self.smooth_l1_beta,
        )

        loss_weight = None
        # loss_weight = torch.ones(pred_class_logits_novel.shape[1]).to(self.device)
        # loss_weight[-1] = 0.1

        return outputs_novel.losses(loss_weight=loss_weight)

