#!/usr/bin/env python
# coding: utf-8

# # Prepare zero-shot split 
# Based on the paper: Bansal, Ankan, et al. "Zero-shot object detection." Proceedings of the European Conference on Computer Vision (ECCV). 2018.

# In[1]:


import json


# In[2]:


import numpy as np


# In[3]:


import torch


# In[19]:


from adet.config import get_cfg
from adet.modeling.language_backbone.transformers import BERT
import pandas as pd
import numpy as np
# In[5]:
# df_stats = pd.read_csv('./datasets/openimages/annotations/openimages_stats_val.csv')
num_unseen = 100

map_name = {'Computer mouse':'Mouse','Studio couch':'studio couch'}
def replace_cat_name(data):
    for idx, item in enumerate(data['categories']):
        if item['name'] in map_name:
            item['name'] = map_name[item['name']]
    return data
# In[6]:

with open('./datasets/openimages/annotations/openimages_challenge_2019_train_mask_expand_fast.json', 'r') as fin:
    openimages_train_anno_all = replace_cat_name(json.load(fin))


# In[7]:


with open('./datasets/openimages/annotations/openimages_challenge_2019_train_mask_expand_fast.json', 'r') as fin:
    openimages_train_anno_seen = replace_cat_name(json.load(fin))


# In[8]:


with open('./datasets/openimages/annotations/openimages_challenge_2019_train_mask_expand_fast.json', 'r') as fin:
    openimages_train_anno_unseen = replace_cat_name(json.load(fin))


# In[9]:


with open('./datasets/openimages/annotations/openimages_challenge_2019_val_mask_expand_fast.json', 'r') as fin:
    openimages_val_anno_all = replace_cat_name(json.load(fin))


# In[10]:


with open('./datasets/openimages/annotations/openimages_challenge_2019_val_mask_expand_fast.json', 'r') as fin:
    openimages_val_anno_seen = replace_cat_name(json.load(fin))


# In[11]:


with open('./datasets/openimages/annotations/openimages_challenge_2019_val_mask_expand_fast.json', 'r') as fin:
    openimages_val_anno_unseen = replace_cat_name(json.load(fin))


# In[ ]:





# In[12]:


with open('./datasets/openimages/zero-shot/openimages_seen_classes_{}.json'.format(num_unseen), 'r') as fin:
    labels_seen = json.load(fin)


# In[13]:


with open('./datasets/openimages/zero-shot/openimages_unseen_classes_{}.json'.format(num_unseen), 'r') as fin:
    labels_unseen = json.load(fin)


# In[14]:


len(labels_seen), len(labels_unseen)


# In[15]:


labels_all = [item['name'] for item in openimages_val_anno_all['categories']]


# In[ ]:





# In[16]:


set(labels_seen) - set(labels_all)


# In[17]:


set(labels_unseen) - set(labels_all)


# In[ ]:





# In[17]:

class_id_to_split = {}
class_name_to_split = {}
for item in openimages_val_anno_all['categories']:
    if item['name'] in labels_seen:
        class_id_to_split[item['id']] = 'seen'
        class_name_to_split[item['name']] = 'seen'
    elif item['name'] in labels_unseen:
        class_id_to_split[item['id']] = 'unseen'
        class_name_to_split[item['name']] = 'unseen'
    else:
        print('Not found {}'.format(item['name']))
        import pdb; pdb.set_trace()

# In[18]:


# class_name_to_glove = {}
# with open('./datasets/coco/zero-shot/glove.6B.300d.txt', 'r') as fin:
#     for row in fin:
#         row_tk = row.split()
#         if row_tk[0] in class_name_to_split:
#             class_name_to_glove[row_tk[0]] = [float(num) for num in row_tk[1:]]


# In[ ]:





# In[19]:

cfg = get_cfg()
bert = BERT(cfg)


# In[20]:


_ = bert.to('cuda')


# In[21]:


# class_name_to_bertemb = {}
# for c in class_name_to_split:
#     if c not in bert.tokenizer.vocab:
#         print(f'{c} not found')
#         continue
#     cid = bert.tokenizer.vocab[c]
#     class_name_to_bertemb[c] = bert.embeddings[cid]

# import pdb; pdb.set_trace()
# In[22]:


class_list = list(class_name_to_split.keys())


# In[23]:
#import pdb; pdb.set_trace()

encoded_class_list = bert(class_list)


# In[24]:


mask = (1 - encoded_class_list['special_tokens_mask']).to(torch.float32)


# In[25]:


mask.sum(-1)


# In[26]:


embeddings = (encoded_class_list['input_embeddings'] * mask[:, :, None]).sum(1) / mask.sum(1)[:, None]      # << the sum would deal with multiple words in a single label


# In[27]:


embeddings = embeddings.cpu().numpy()


# In[28]:


embeddings.shape


# In[29]:


class_name_to_bertemb = {}
for c, emb in zip(class_list, embeddings.tolist()):
    class_name_to_bertemb[c] = emb


# In[31]:


def filter_annotation(anno_dict, split_name_list):
    filtered_categories = []
    for item in anno_dict['categories']:
        if class_id_to_split.get(item['id']) in split_name_list:
            item['embedding'] = {}
            #item['embedding']['GloVE'] = class_name_to_glove[item['name']]
            item['embedding']['BertEmb'] = class_name_to_bertemb[item['name']]
            item['split'] = class_id_to_split.get(item['id'])
            filtered_categories.append(item)
    anno_dict['categories'] = filtered_categories
    
    filtered_images = []
    filtered_annotations = []
    useful_image_ids = set()
    for item in anno_dict['annotations']:
        if class_id_to_split.get(item['category_id']) in split_name_list:
            filtered_annotations.append(item)
            useful_image_ids.add(item['image_id'])
    for item in anno_dict['images']:
        if item['id'] in useful_image_ids:
            filtered_images.append(item)
    anno_dict['annotations'] = filtered_annotations
    anno_dict['images'] = filtered_images    


# In[ ]:





# In[32]:


filter_annotation(openimages_train_anno_seen, ['seen'])


# In[33]:


filter_annotation(openimages_train_anno_unseen, ['unseen'])


# In[34]:


filter_annotation(openimages_train_anno_all, ['seen', 'unseen'])


# In[35]:


filter_annotation(openimages_val_anno_seen, ['seen'])


# In[36]:


filter_annotation(openimages_val_anno_unseen, ['unseen'])


# In[37]:


filter_annotation(openimages_val_anno_all, ['seen', 'unseen'])


# In[38]:


len(openimages_val_anno_seen['categories']), len(openimages_val_anno_unseen['categories']), len(openimages_val_anno_all['categories'])


# In[ ]:





# In[39]:


with open('./datasets/openimages/zero-shot/instances_train2019_mask_seen_{}.json'.format(num_unseen), 'w') as fout:
    json.dump(openimages_train_anno_seen, fout)


# In[40]:


with open('./datasets/openimages/zero-shot/instances_train2019_mask_unseen_{}.json'.format(num_unseen), 'w') as fout:
    json.dump(openimages_train_anno_unseen, fout)


# In[41]:


with open('./datasets/openimages/zero-shot/__instances_train2019_mask_all_{}.json'.format(num_unseen), 'w') as fout:
    json.dump(openimages_train_anno_all, fout)


# In[42]:


with open('./datasets/openimages/zero-shot/instances_val2019_mask_seen_{}.json'.format(num_unseen), 'w') as fout:
    json.dump(openimages_val_anno_seen, fout)


# In[43]:


with open('./datasets/openimages/zero-shot/instances_val2019_mask_unseen_{}.json'.format(num_unseen), 'w') as fout:
    json.dump(openimages_val_anno_unseen, fout)


# In[44]:


with open('./datasets/openimages/zero-shot/instances_val2019_mask_all_{}.json'.format(num_unseen), 'w') as fout:
    json.dump(openimages_val_anno_all, fout)


# In[ ]: