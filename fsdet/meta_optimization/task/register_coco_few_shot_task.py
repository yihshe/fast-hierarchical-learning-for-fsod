"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.
We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations
We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Here we only register the few-shot datasets and complete COCO, PascalVOC and 
LVIS have been handled by the builtin datasets in detectron2. 
"""
import io
import os
import contextlib
import numpy as np

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.data.datasets.lvis import (
    get_lvis_instances_meta,
    register_lvis_instances,
)

from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO
import torch
import random

from IPython import embed

# ==== Predefined datasets and splits for COCO ==========

def register_all_coco_tasks(num_tasks=1000, root="datasets"):
    random.seed(0)

    META_TASKS = list([])
    shots = {'support': 5, 'query': 15}
    COCO_BASE_CATEGORIES = torch.load('datasets/cocosplit/datasplit/coco_base_categories.pt')
    category_dict = {C['id']: C for C in COCO_BASE_CATEGORIES}

    imgdir = "coco/trainval2014"

    for i in range(num_tasks):
        metadata = _generate_task_metadata(category_dict)
        for task_set in ['support', 'query']:
            name = "coco_trainval_task{}_{}_{}shot".format(i, task_set, shots[task_set])
            annofile = "coco_tasks/task{}/{}".format(i, task_set)

            register_single_coco_task(
                name,
                metadata, 
                os.path.join(root, imgdir),
                os.path.join(root, annofile),
            )

def _generate_task_metadata(category_dict:dict, num_novel_classes = 20):
    thing_ids = list(category_dict.keys())
    assert len(thing_ids) == 60, len(thing_ids)

    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [category_dict[k]["name"] for k in thing_ids]
    thing_colors = [category_dict[k]["color"] for k in thing_ids]
    
    novel_ids = list(sorted(random.sample(thing_ids, num_novel_classes)))
    novel_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(novel_ids)}
    novel_classes = [category_dict[k]["name"] for k in novel_ids]

    base_ids = list([i for i in thing_ids if i not in novel_ids])
    base_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(base_ids)}
    base_classes = [category_dict[k]["name"] for k in base_ids]
    
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,

        "novel_dataset_id_to_contiguous_id": novel_dataset_id_to_contiguous_id,
        "novel_classes": novel_classes,

        "base_dataset_id_to_contiguous_id": base_dataset_id_to_contiguous_id,
        "base_classes": base_classes,
        }
    
    return ret

def register_single_coco_task(name, metadata, imgdir, annofile):
    DatasetCatalog.register(
        name,
        lambda: load_coco_json(annofile, imgdir, metadata, name),
    )
    # TODO use the split info in metadata to randomly initialize the weights (with seed 0)
    MetadataCatalog.get(name).set(
        # json_file = ""
        json_file=annofile,
        image_root=imgdir,
        evaluator_type="coco",
        dirname="datasets/coco",
        **metadata,
    )

def load_coco_json(annofile, image_root, metadata, dataset_name):
    fileids = {}
    shot = dataset_name.split("_")[-1].split("shot")[0]
    for idx, cls in enumerate(metadata["thing_classes"]):
        json_file = os.path.join(
            annofile, "full_box_{}shot_{}_trainval.json".format(shot, cls)
        )
        json_file = PathManager.get_local_path(json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            coco_api = COCO(json_file)

        img_ids = sorted(list(coco_api.imgs.keys()))
        imgs = coco_api.loadImgs(img_ids)
        anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
        fileids[idx] = list(zip(imgs, anns))

    id_map = metadata["thing_dataset_id_to_contiguous_id"]

    dataset_dicts = []
    ann_keys = ["iscrowd", "bbox", "category_id"]

    for _, fileids_ in fileids.items():
        dicts = []
        for (img_dict, anno_dict_list) in fileids_:
            for anno in anno_dict_list:
                record = {}
                record["file_name"] = os.path.join(
                    image_root, img_dict["file_name"]
                )
                record["height"] = img_dict["height"]
                record["width"] = img_dict["width"]
                image_id = record["image_id"] = img_dict["id"]

                assert anno["image_id"] == image_id
                assert anno.get("ignore", 0) == 0

                obj = {key: anno[key] for key in ann_keys if key in anno}

                obj["bbox_mode"] = BoxMode.XYWH_ABS
                # TODO notice the transfer from category id to continuous id
                # one record for one annotation, as a dict in the list 'dicts'
                obj["category_id"] = id_map[obj["category_id"]]
                record["annotations"] = [obj]

                dicts.append(record)
        if len(dicts) > int(shot):
            dicts = np.random.choice(dicts, int(shot), replace=False)
        # the dicts of different categories (file_ids) will all be added to dataset_dicts
        dataset_dicts.extend(dicts)

    return dataset_dicts

# Register them all under "./datasets"
register_all_coco_tasks()