from IPython.terminal.embed import embed
import numpy as np
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO

import contextlib
import io
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

from .meta_coco_hda import HDAMetaInfo

"""
This file contains functions to parse COCO-format annotations into dicts in "Detectron2 format".
"""


__all__ = ["register_meta_coco"]


def load_coco_json(json_file, image_root, metadata, dataset_name):
    """
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection.
    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str): the directory where the images in this json file exists.
        metadata: meta data associated with dataset_name
        dataset_name (str): the name of the dataset (e.g., coco_2017_train).
            If provided, this function will also put "thing_classes" into
            the metadata associated with this dataset.
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    is_shots = "shot" in dataset_name
    
    if is_shots:
        fileids = {}
        split_dir = os.path.join("datasets", "cocosplit")
        # TODO if the random seed needs to be specified when sampling shots
        if "seed" in dataset_name:
            shot = dataset_name.split("_")[-2].split("shot")[0]
            seed = int(dataset_name.split("_seed")[-1])
            split_dir = os.path.join(split_dir, "seed{}".format(seed))
        else:
            shot = dataset_name.split("_")[-1].split("shot")[0]
        # TODO the thing_classes needs to be shrinked to 60
        # and the psudo novel and base classes in the base set needs to be randomly splitted for each task
        # TODO HDA, for few-shot dataset, the selecttion of corresponding classes happens here, stored in fileids, each entry coresponds to a class
        for idx, cls in enumerate(metadata["thing_classes"]):
            # NOTE if thing_classes is super class, then include all its child class at a given order
            json_file = os.path.join(
                split_dir, "full_box_{}shot_{}_trainval.json".format(shot, cls)
            )
            json_file = PathManager.get_local_path(json_file)
            with contextlib.redirect_stdout(io.StringIO()):
                coco_api = COCO(json_file)
            img_ids = sorted(list(coco_api.imgs.keys()))
            imgs = coco_api.loadImgs(img_ids)
            anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
            fileids[idx] = list(zip(imgs, anns))
    else:
        json_file = PathManager.get_local_path(json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            coco_api = COCO(json_file)
        # sort indices for reproducible results
        img_ids = sorted(list(coco_api.imgs.keys()))
        imgs = coco_api.loadImgs(img_ids)
        # NOTE anns here is only the annotation part of the json file
        anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
        imgs_anns = list(zip(imgs, anns))

    id_map = metadata["thing_dataset_id_to_contiguous_id"]

    dataset_dicts = []
    ann_keys = ["iscrowd", "bbox", "category_id"]

    if is_shots:
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

                    # NOTE HDA the category id here is still the id of child class
                    obj = {key: anno[key] for key in ann_keys if key in anno}

                    obj["bbox_mode"] = BoxMode.XYWH_ABS
                    # TODO notice the transfer from category id to continuous id
                    # one record for one annotation, as a dict in the list 'dicts'
                    # TODO HDA, map of class id used in gt, category_id of child -> super_category_id 
                    # -> corrsponding continuous id that can be used as gt, also change the thing_classes to contain super class
                    # NOTE even in fewshot dataset no need for mapping the child id to parent id, the id_map needs revision to map it to contiguous id correctly
                    obj["category_id"] = id_map[obj["category_id"]]
                    record["annotations"] = [obj]
                    dicts.append(record)
            if len(dicts) > int(shot):
                dicts = np.random.choice(dicts, int(shot), replace=False)
            # the dicts of different categories (file_ids) will all be added to dataset_dicts
            dataset_dicts.extend(dicts)
    else:
        for (img_dict, anno_dict_list) in imgs_anns:
            record = {}
            record["file_name"] = os.path.join(
                image_root, img_dict["file_name"]
            )
            record["height"] = img_dict["height"]
            record["width"] = img_dict["width"]
            image_id = record["image_id"] = img_dict["id"]

            objs = []
            for anno in anno_dict_list:
                assert anno["image_id"] == image_id
                assert anno.get("ignore", 0) == 0

                obj = {key: anno[key] for key in ann_keys if key in anno}

                obj["bbox_mode"] = BoxMode.XYWH_ABS

                if obj["category_id"] in id_map:
                    obj["category_id"] = id_map[obj["category_id"]]
                    # difference, every recorde corresponds to one image with several annotations
                    objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)

    return dataset_dicts

def register_meta_coco(name, metadata, imgdir, annofile):
    DatasetCatalog.register(
        name,
        lambda: load_coco_json(annofile, imgdir, metadata, name),
    )

    if "_base" in name or "_novel" in name:
        split = "base" if "_base" in name else "novel"
        metadata["thing_dataset_id_to_contiguous_id"] = metadata[
            "{}_dataset_id_to_contiguous_id".format(split)
        ]
        metadata["thing_classes"] = metadata["{}_classes".format(split)]

    MetadataCatalog.get(name).set(
        json_file=annofile,
        image_root=imgdir,
        evaluator_type="coco",
        dirname="datasets/coco",
        **metadata,
    )