from maskrcnn_benchmark.data.datasets.conceptual_captions import ConCapDataset
import torch
from tqdm import tqdm
import pickle
import os, sys

img_dir = "/trainman-mount/trainman-storage-add87537-4c73-4d36-9774-0fddd6d5fd4f/datasets/conceptual/images/"
anno_dir = "/trainman-mount/trainman-storage-add87537-4c73-4d36-9774-0fddd6d5fd4f/datasets/conceptual/img_lists/"

id_partition = int(sys.argv[1])
n_partition = int(sys.argv[2])

split = "train"
dataset = ConCapDataset(split=split,img_dir=img_dir, anno_dir=anno_dir)
meta_dic = {}
for idx in tqdm(range(len(dataset))):
    if idx % n_partition != id_partition:
        continue
    img, target, idx = dataset[idx]
    width, height = img.size
    meta_dic[idx] = {'height':height,'width':width}
    # import pdb; pdb.set_trace()

with open('./datasets/conceptual/meta_{}_id_part_{}_{}.pkl'.format(split,id_partition,n_partition),'wb') as f:
    pickle.dump(meta_dic,f)

split = "val"
dataset = ConCapDataset(split=split,img_dir=img_dir, anno_dir=anno_dir)
meta_dic = {}
for idx in tqdm(range(len(dataset))):
    if idx % n_partition != id_partition:
        continue
    img, target, idx = dataset[idx]
    width, height = img.size
    meta_dic[idx] = {'height':height,'width':width}
    # import pdb; pdb.set_trace()

with open('./datasets/conceptual/meta_{}_id_part_{}_{}.pkl'.format(split,id_partition,n_partition),'wb') as f:
    pickle.dump(meta_dic,f)