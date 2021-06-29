import torch
from detectron2.data.build import get_detection_dataset_dicts
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from fsdet.optimization.gradient_mask import GradientMask
import copy
import json
import random
import os
from pycocotools.coco import COCO
from collections import defaultdict
import time
import numpy as np
from IPython.terminal.embed import embed
# finish the whole framework today, with the meta cg trainer being simplified, and prepare uncertainties to discuss

class COCO_API(COCO):
    def __init__(self, annofile: dict):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset,self.anns,self.cats,self.imgs = dict(),dict(),dict(),dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        self.dataset = annofile
        self.createIndex()
    
    def createIndex(self):
        # create index
        # print('creating index...')
        anns, cats, imgs = {}, {}, {}
        imgToAnns,catToImgs = defaultdict(list),defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        # print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats

class TaskSampler():
    def __init__(self, cfg):
        self.num_tasks = cfg.META_PARAMS.TASKS_PER_BATCH
        self.support_set_shots = cfg.META_PARAMS.SUPPORT_SET_SHOTS
        self.query_set_shots = cfg.META_PARAMS.QUERY_SET_SHOTS
        
        # cfg.META_PARAMS.BASE_CATEGORIES_PATH = 'datasets/cocosplit/datasplit/coco_base_categories.pt'
        COCO_BASE_CATEGORIES = list(torch.load(cfg.META_PARAMS.BASE_CATEGORIES_PATH))
        ID2CLASS = {C['id']: C['name'] for C in COCO_BASE_CATEGORIES}
        CLASS2ID = {v: k for k, v in ID2CLASS.items()}
        
        category_dict = {C['id']: C for C in COCO_BASE_CATEGORIES}

        # self.cfg.META_PARAMS.TRAINING_DATA_PATH = 'datasets/cocosplit/datasplit/trainvalno5k.json'
        data = json.load(open(cfg.META_PARAMS.TRAINING_DATA_PATH))
        
        new_all_cats = []
        for cat in data['categories']:
            if cat['id'] in ID2CLASS.keys():
                new_all_cats.append(cat)

        id2img = {}
        for i in data['images']:
            id2img[i['id']] = i
        
        anno = {i: [] for i in ID2CLASS.keys()}
        for a in data['annotations']:
            if a['iscrowd'] == 1 or a['category_id'] not in ID2CLASS.keys():
                continue
            anno[a['category_id']].append(a)
        
        img_ids_all = {}
        for c in ID2CLASS.keys():
            img_ids_all[c] = {}
            for a in anno[c]:
                if a['image_id'] in img_ids_all[c]:
                    img_ids_all[c][a['image_id']].append(a)
                else:
                    img_ids_all[c][a['image_id']] = [a]

        self.ID2CLASS = ID2CLASS
        self.CLASS2ID = CLASS2ID
        self.category_dict = category_dict
        self.data = data
        self.new_all_cats = new_all_cats
        self.id2img = id2img
        self.anno = anno
        self.img_ids_all = img_ids_all
        self.cfg = cfg

        random.seed(0)

    def generate_tasks(self, scope='all'):
        assert scope in ['all', 'base', 'novel']
        meta_tasks = {}
        for i in range(self.num_tasks):
            meta_tasks['task{}'.format(i)] = {}
            # meta_tasks['task{}'.format(i)]['metadata'] 
            metadata = self._generate_task_metadata(self.category_dict)
            if scope == 'novel' or 'scope' == 'base':
                metadata['thing_dataset_id_to_contiguous_id'] = metadata['{}_dataset_id_to_contiguous_id'.format(scope)]
                metadata['thing_classes']=metadata['{}_classes'.format(scope)]
            meta_tasks['task{}'.format(i)]['metadata'] = metadata
            self.generate_single_task(meta_tasks['task{}'.format(i)])
        
        return meta_tasks

    def generate_single_task(self, single_task: dict):
        for x in ['support', 'query']:
            single_task[x] = {}

        # TODO replace ID2CLASS here to sample the task with only novel or base classes
        # for c in self.ID2CLASS.keys():
        for c in single_task['metadata']['thing_dataset_id_to_contiguous_id'].keys():
            img_ids = self.img_ids_all[c]
            support_set, sampled_img_ids = self.sampling_shots(img_ids, self.support_set_shots, self.id2img, self.data, self.new_all_cats)
            query_set, sampled_img_ids_query = self.sampling_shots(img_ids, self.query_set_shots, self.id2img, self.data, self.new_all_cats,
                                            img_ids_exclude=sampled_img_ids)
            self.store_shots(single_task['support'], support_set, self.ID2CLASS[c], self.support_set_shots)
            self.store_shots(single_task['query'], query_set, self.ID2CLASS[c], self.query_set_shots)
            
    @classmethod
    def sampling_shots(cls, img_ids, shots, id2img, data, new_all_cats, img_ids_exclude = None):
        sample_shots = []
        sample_imgs = []
        sampled_img_ids = []

        imgs_to_sample = list(img_ids.keys()) if img_ids_exclude is None else list([i for i in img_ids.keys() if i not in img_ids_exclude])

        while True:
            ids = random.sample(imgs_to_sample, shots)
            for id in ids:
                skip = False
                for s in sample_shots:
                    if id == s['image_id']:
                        skip = True
                        break
                if skip:
                    continue
                if len(img_ids[id]) + len(sample_shots) > shots:
                    continue
                sample_shots.extend(img_ids[id])
                sample_imgs.append(id2img[id])
                sampled_img_ids.append(id)
                if len(sample_shots) == shots:
                    break

            if len(sample_shots) == shots:
                break

        new_data = {
            'info': data['info'],
            'licenses': data['licenses'],
            'categories': new_all_cats,
            'images': sample_imgs,
            'annotations': sample_shots
        }

        return new_data, sampled_img_ids

    @classmethod
    def store_shots(cls, task_set: dict, new_data, category, shots):
        anno_name = 'full_box_{}shot_{}_trainval'.format(shots, category)
        task_set[anno_name] = new_data

    def register_tasks(self, meta_tasks, root = "datasets"):
        task_set_names = []
        # register the sampled meta tasks at each batch
        shots = {'support': self.support_set_shots, 'query': self.query_set_shots}
        imgdir = "coco/trainval2014"
        for i in range(len(meta_tasks)):
            task_set_name = {}
            # metadata = self._generate_task_metadata(self.category_dict)
            metadata = meta_tasks['task{}'.format(i)]['metadata']
            for x in ['support', 'query']:
                name = "coco_trainval_task{}_{}_{}shot".format(i, x, shots[x])
                annofiles = meta_tasks['task{}'.format(i)][x]
                task_set_name[x] = name
                self.register_single_coco_task(
                    name, 
                    metadata,
                    os.path.join(root, imgdir),
                    # the annofile here is a dict, not a path
                    annofiles,
                )
            task_set_names.append(task_set_name)

        return task_set_names

    def register_single_coco_task(self, name, metadata, imgdir, annofile):
        if name in DatasetCatalog:
            DatasetCatalog.remove(name)
            MetadataCatalog.remove(name)
        DatasetCatalog.register(
            name,
            lambda: self.load_coco_json(annofile, imgdir, metadata, name),
        )
        MetadataCatalog.get(name).set(
            json_file = "",
            image_root=imgdir,
            evaluator_type="coco",
            dirname="datasets/coco",
            **metadata,
        )

    @classmethod
    def _generate_task_metadata(cls, category_dict:dict, num_novel_classes=20):
        thing_ids = list(category_dict.keys())
        assert len(thing_ids) == 60, len(thing_ids)

        thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
        thing_classes = [category_dict[k]["name"] for k in thing_ids]
        # thing_colors = [category_dict[k]["color"] for k in thing_ids]
        
        novel_ids = list(sorted(random.sample(thing_ids, num_novel_classes)))
        novel_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(novel_ids)}
        novel_classes = [category_dict[k]["name"] for k in novel_ids]

        base_ids = list([i for i in thing_ids if i not in novel_ids])
        base_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(base_ids)}
        base_classes = [category_dict[k]["name"] for k in base_ids]
        
        # TODO overwirte the thing dataset id accordingly for only novel for old tasks
        ret = {
            "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
            "thing_classes": thing_classes,
            # "thing_colors": thing_colors,

            "novel_dataset_id_to_contiguous_id": novel_dataset_id_to_contiguous_id,
            "novel_classes": novel_classes,

            "base_dataset_id_to_contiguous_id": base_dataset_id_to_contiguous_id,
            "base_classes": base_classes,
            }
        
        return ret

    @classmethod
    def load_coco_json(cls, annofiles, image_root, metadata, dataset_name):
        fileids = {}
        shot = dataset_name.split("_")[-1].split("shot")[0]
        for idx, category in enumerate(metadata["thing_classes"]):
            annofile = annofiles["full_box_{}shot_{}_trainval".format(shot, category)]
            coco_api = COCO_API(annofile)
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

# build custom task set and loader in this script
class TaskDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, pretrained_params: dict = None):
        self.cfg = cfg
        # here the pretrained_params is the full state dict of the pretrained model
        self.pretrained_params = pretrained_params

    def build_tasks(self, task_set_names, meta_task_type = 'coco_meta_all_tfa'):
        assert meta_task_type in ['coco_meta_all_tfa', 'coco_meta_novel']
        tasks = []
        for i in range(len(task_set_names)):
            tasks.append(self.build_single_task(task_set_names[i], meta_task_type=meta_task_type))
        return tasks

    def build_single_task(self, task_set_name, meta_task_type = 'coco_meta_all_tfa'):
        # return the suport set, query set, cfg, gradient mask, new weights and base params 
        task = {}
        task['data'] = {}
        for task_set in ['support', 'query']:
            task['data'][task_set] = self.get_dataset(self.cfg, task_set_name[task_set])

        # can be moved to other part later on to generate it one time and store it with the task
        metadata = MetadataCatalog.get(task_set_name['support'])
        setup_generator = GradientMask(meta_task_type, metadata, self.pretrained_params)
        task['setup'] = setup_generator.create_meta_setup()

        return task
    
    @classmethod
    def get_dataset(cls, cfg, dataset_name):
        dataset = get_detection_dataset_dicts(
            (dataset_name,),
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
        )
        return dataset