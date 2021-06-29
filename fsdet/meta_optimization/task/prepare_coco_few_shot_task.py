import argparse
import json
import os
import random
from IPython import embed
import torch
"""
This script is used to sample the few-shot tasks for meta learning
"""

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--seeds", type=int, nargs="+", default=[1, 10],
    #                     help="Range of seeds")
    parser.add_argument("--num_tasks", type=int, default=1000,
                        help="Number of tasks to sample")
    parser.add_argument("--support_set_shots", type=int, default=5,
                        help="Size of support set")
    parser.add_argument("--query_set_shots", type=int, default=15,
                        help="Size of query set")
    args = parser.parse_args()
    return args

def generate_tasks(args):
    # generate the task path and iterate over the total task number
    # then use it to generate the task, randomly split the classes and register the dataset
    data_path = 'datasets/cocosplit/datasplit/trainvalno5k.json'
    data = json.load(open(data_path))

    new_all_cats = []
    for cat in data['categories']:
        if cat['id'] in ID2CLASS.keys():
            new_all_cats.append(cat)

    id2img = {}
    for i in data['images']:
        id2img[i['id']] = i
    
    # the anno now should only contains the category id and associated annotations for 60 base classes
    anno = {i: [] for i in ID2CLASS.keys()}
    for a in data['annotations']:
        if a['iscrowd'] == 1 or a['category_id'] not in ID2CLASS.keys():
            continue
        anno[a['category_id']].append(a)
    
    for i in range(args.num_tasks):
        task_path = 'datasets/coco_tasks/task{}'.format(i)
        generate_single_task(args, data, new_all_cats, id2img, anno, task_path)
        print('task {}'.format(i))

    print('done')
    

def generate_single_task(args, data, new_all_cats, id2img, anno, task_path):
    for c in ID2CLASS.keys():
        img_ids = {}
        for a in anno[c]:
            if a['image_id'] in img_ids:
                img_ids[a['image_id']].append(a)
            else:
                img_ids[a['image_id']] = [a]

        support_set, sampled_img_ids = sampling_shots(img_ids, args.support_set_shots, id2img, data, new_all_cats)
        query_set, sampled_img_ids_query = sampling_shots(img_ids, args.query_set_shots, id2img, data, new_all_cats,
                                        img_ids_exclude=sampled_img_ids)
        
        save_shots(support_set, task_path, ID2CLASS[c], args.support_set_shots)
        save_shots(query_set, task_path, ID2CLASS[c], args.query_set_shots, query=True)

# for shots in [1, 2, 3, 5, 10, 30]:
def sampling_shots(img_ids, shots, id2img, data, new_all_cats, img_ids_exclude = None):
    sample_shots = []
    sample_imgs = []
    sampled_img_ids = []

    imgs_to_sample = list(img_ids.keys()) if img_ids_exclude is None else list([i for i in img_ids.keys() if i not in img_ids_exclude])

    while True:
        # TODO store the image_id (img) that has been used for support set
        # and exclude it from the keys of the query set
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

def save_shots(new_data, task_path, category, shots, query = False):
    save_path = get_save_path_sets(task_path, category, shots, query)
    with open(save_path, 'w') as f:
        json.dump(new_data, f)

def get_save_path_sets(task_path, category, shots, query):
    prefix = 'full_box_{}shot_{}_trainval'.format(shots, category)
    save_dir = os.path.join(task_path, 'query' if query else 'support')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, prefix + '.json')
    return save_path


if __name__ == '__main__':
    # load the 60 base classes contained in the base set
    COCO_BASE_CATEGORIES = torch.load('datasets/cocosplit/datasplit/coco_base_categories.pt')
    ID2CLASS = {C['id']: C['name'] for C in COCO_BASE_CATEGORIES}
    CLASS2ID = {v: k for k, v in ID2CLASS.items()}
    args = parse_args()
    generate_tasks(args)
