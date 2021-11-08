import copy
from detectron2.data import detection_utils as utils 
from detectron2.config import configurable
from detectron2.data.samplers.distributed_sampler import InferenceSampler
from detectron2.utils.comm import get_world_size
import numpy as np
import torch
import operator
from detectron2.data.build import get_detection_dataset_dicts, trivial_batch_collator, worker_init_reset_seed, _train_loader_from_config
from detectron2.data.common import AspectRatioGroupedDataset, DatasetFromList, MapDataset
from detectron2.data.dataset_mapper import DatasetMapper
import logging
from detectron2.data.samplers import RepeatFactorTrainingSampler, TrainingSampler
from torch.utils.data.sampler import SequentialSampler
from IPython.terminal.embed import embed

from detectron2.data import transforms as T

def build_batch_data_loader(
    dataset, sampler, total_batch_size, *, aspect_ratio_grouping=False, num_workers=0, extract_features = False
):
    """
    Build a batched dataloader for training.
    Args:
        dataset (torch.utils.data.Dataset): map-style PyTorch dataset. Can be indexed.
        sampler (torch.utils.data.sampler.Sampler): a sampler that produces indices
        total_batch_size, aspect_ratio_grouping, num_workers): see
            :func:`build_detection_train_loader`.
    Returns:
        iterable[list]. Length of each list is the batch size of the current
            GPU. Each element in the list comes from the dataset.
    """
    world_size = get_world_size()
    assert (
        total_batch_size > 0 and total_batch_size % world_size == 0
    ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size, world_size
    )

    batch_size = total_batch_size // world_size
    if aspect_ratio_grouping:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            num_workers=num_workers,
            batch_sampler=None,
            collate_fn=operator.itemgetter(0),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
        )  # yield individual mapped dict
        return AspectRatioGroupedDataset(data_loader, batch_size)
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, batch_size, drop_last=False if extract_features else True
        )  # drop_last so the batch always have the same size
        return torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=trivial_batch_collator,
            worker_init_fn=worker_init_reset_seed,
        )

def _train_loader_from_config(cfg, mapper=None, *, dataset=None, sampler=None, extract_features=False, mapped=False):
    if dataset is None:
        dataset = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
        )
        # _log_api_usage("dataset." + cfg.DATASETS.TRAIN[0])

    if mapper is None and mapped is False:
        mapper = DatasetMapper(cfg, True)

    if sampler is None:
        if extract_features:
            sampler = SequentialSampler(dataset)
        else:
            sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
            logger = logging.getLogger(__name__)
            logger.info("Using training sampler {}".format(sampler_name))
            if sampler_name == "TrainingSampler":
                sampler = TrainingSampler(len(dataset))
            elif sampler_name == "RepeatFactorTrainingSampler":
                repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
                    dataset, cfg.DATALOADER.REPEAT_THRESHOLD
                )
                sampler = RepeatFactorTrainingSampler(repeat_factors)
            else:
                raise ValueError("Unknown training sampler: {}".format(sampler_name))
    
    return {
        "dataset": dataset,
        "sampler": sampler,
        "mapper": mapper,
        "total_batch_size": cfg.SOLVER.IMS_PER_BATCH_FEAT_EXTRACT if extract_features else cfg.SOLVER.IMS_PER_BATCH,
        "aspect_ratio_grouping": cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        "num_workers": cfg.DATALOADER.NUM_WORKERS if extract_features else 0,
        "extract_features": extract_features,
    }

@configurable(from_config=_train_loader_from_config)
def build_detection_train_loader(
    dataset, *, mapper, sampler=None, total_batch_size, aspect_ratio_grouping=True, num_workers=0, extract_features = False
):
    """
    Build a dataloader for object detection with some default features.
    This interface is experimental.
    Args:
        dataset (list or torch.utils.data.Dataset): a list of dataset dicts,
            or a map-style pytorch dataset. They can be obtained by using
            :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper (callable): a callable which takes a sample (dict) from dataset and
            returns the format to be consumed by the model.
            When using cfg, the default choice is ``DatasetMapper(cfg, is_train=True)``.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that produces
            indices to be applied on ``dataset``. Default to :class:`TrainingSampler`,
            which coordinates an infinite random shuffle sequence across all workers.
        total_batch_size (int): total batch size across all workers. Batching
            simply puts data into a list.
        aspect_ratio_grouping (bool): whether to group images with similar
            aspect ratio for efficiency. When enabled, it requires each
            element in dataset be a dict with keys "width" and "height".
        num_workers (int): number of parallel data loading workers
    Returns:
        torch.utils.data.DataLoader: a dataloader. Each output from it is a
            ``list[mapped_element]`` of length ``total_batch_size / num_workers``,
            where ``mapped_element`` is produced by the ``mapper``.
    """
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    if sampler is None:
        sampler = TrainingSampler(len(dataset))
    assert isinstance(sampler, torch.utils.data.sampler.Sampler)
    return build_batch_data_loader(
        dataset,
        sampler,
        total_batch_size,
        aspect_ratio_grouping=aspect_ratio_grouping,
        num_workers=num_workers,
        extract_features = extract_features
    )

def _test_loader_from_config_with_gt(cfg, dataset_name, mapper=None):
    """
    Uses the given `dataset_name` argument (instead of the names in cfg), because the
    standard practice is to evaluate each test set individually (not combining them).
    """
    dataset = get_detection_dataset_dicts(
        [dataset_name],
        filter_empty=False,
        proposal_files=[
            cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(dataset_name)]
        ]
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )
    if mapper is None:
        mapper = DatasetMapper_with_gt(cfg, False)
    return {"dataset": dataset, "mapper": mapper, "num_worker": cfg.DATALOADER.NUM_WORKERS}


@configurable(from_config=_test_loader_from_config_with_gt)
def build_detection_test_loader_with_gt(dataset, *, mapper, num_worker=0):
    """
    Similar to `build_detection_train_loader`, but uses a batch size of 1.
    This interface is experimental.

    Args:
        dataset (list or torch.utils.data.Dataset): a list of dataset dicts,
            or a map-style pytorch dataset. They can be obtained by using
            :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper (callable): a callable which takes a sample (dict) from dataset
           and returns the format to be consumed by the model.
           When using cfg, the default choice is ``DatasetMapper(cfg, is_train=False)``.
        num_workers (int): number of parallel data loading workers

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.

    Examples:
    ::
        data_loader = build_detection_test_loader(
            DatasetRegistry.get("my_test"),
            mapper=DatasetMapper(...))

        # or, instantiate with a CfgNode:
        data_loader = build_detection_test_loader(cfg, "my_test")
    """
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    sampler = InferenceSampler(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_worker,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader

class DatasetMapper_with_gt(DatasetMapper):
    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        # if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            # dataset_dict.pop("annotations", None)
            # dataset_dict.pop("sem_seg_file_name", None)

            # return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)
            
            if not self.is_train:
                annos = [obj for obj in dataset_dict.pop("annotations") if obj.get("iscrowd", 0) == 0]
                instances = utils.annotations_to_instances(
                    annos, image_shape, mask_format=self.instance_mask_format
                )
                dataset_dict["instances"] = utils.filter_empty_instances(instances)
            
            else:
                # USER: Implement additional transformations if you have other types of data
                annos = [
                    utils.transform_instance_annotations(
                        obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                    )
                    for obj in dataset_dict.pop("annotations")
                    if obj.get("iscrowd", 0) == 0
                ]
                instances = utils.annotations_to_instances(
                    annos, image_shape, mask_format=self.instance_mask_format
                )

                # After transforms such as cropping are applied, the bounding box may no longer
                # tightly bound the object. As an example, imagine a triangle object
                # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
                # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
                # the intersection of original bounding box and the cropping box.
                if self.recompute_boxes:
                    instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
                dataset_dict["instances"] = utils.filter_empty_instances(instances)
                
        return dataset_dict



class ExtractedDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, proposals, box_features):
        'Initialization'
        self.proposals = proposals
        self.box_features = box_features
        # TODO check later if this applies to the other dataset (same num of instances per img)
        # self.feat_dim = int(box_features.shape[0]/len(proposals))
        self.num_preds_per_image = torch.tensor([len(proposal) for proposal in proposals]).long()

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.proposals)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        proposal = self.proposals[index]
        # box_feature = self.box_features[index*self.feat_dim:(index+1)*self.feat_dim, :]
        box_feature = self.box_features[torch.sum(self.num_preds_per_image[:index]):(torch.sum(self.num_preds_per_image[:index])+self.num_preds_per_image[index])]

        return proposal, box_feature