import pickle
import os, sys
#%%

n_partition = 40
template = './datasets/conceptual/meta_{}_id_part_{}_{}.pkl'

split = 'train'
all_meta_dic = {}
for id_partition in range(n_partition):
    with open('./datasets/conceptual/meta_{}_id_part_{}_{}.pkl'.format(split,id_partition,n_partition),'rb') as f:
        meta_dic = pickle.load(f)
        all_meta_dic.update(meta_dic)

with open('./datasets/conceptual/all_meta_{}.pkl'.format(split),'wb') as f:
    pickle.dump(all_meta_dic,f)

import pdb; pdb.set_trace()
#%%
split = 'val'
all_meta_dic = {}
for id_partition in range(n_partition):
    with open('./datasets/conceptual/meta_{}_id_part_{}_{}.pkl'.format(split,id_partition,n_partition),'rb') as f:
        meta_dic = pickle.load(f)
        all_meta_dic.update(meta_dic)

with open('./datasets/conceptual/all_meta_{}.pkl'.format(split),'wb') as f:
    pickle.dump(all_meta_dic,f)

import pdb; pdb.set_trace()