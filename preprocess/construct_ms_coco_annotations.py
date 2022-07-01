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


from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.language_backbone.transformers import BERT


# In[5]:





# In[6]:


with open('../datasets/coco/annotations/instances_train2017.json', 'r') as fin:
    coco_train_anno_all = json.load(fin)


# In[7]:


with open('../datasets/coco/annotations/instances_train2017.json', 'r') as fin:
    coco_train_anno_seen = json.load(fin)


# In[8]:


with open('../datasets/coco/annotations/instances_train2017.json', 'r') as fin:
    coco_train_anno_unseen = json.load(fin)


# In[9]:


with open('../datasets/coco/annotations/instances_val2017.json', 'r') as fin:
    coco_val_anno_all = json.load(fin)


# In[10]:


with open('../datasets/coco/annotations/instances_val2017.json', 'r') as fin:
    coco_val_anno_seen = json.load(fin)


# In[11]:


with open('../datasets/coco/annotations/instances_val2017.json', 'r') as fin:
    coco_val_anno_unseen = json.load(fin)


# In[ ]:





# In[12]:


with open('../datasets/coco/zero-shot/mscoco_seen_classes.json', 'r') as fin:
    labels_seen = json.load(fin)


# In[13]:


with open('../datasets/coco/zero-shot/mscoco_unseen_classes.json', 'r') as fin:
    labels_unseen = json.load(fin)


# In[14]:


len(labels_seen), len(labels_unseen)


# In[15]:


labels_all = [item['name'] for item in coco_val_anno_all['categories']]


# In[ ]:





# In[16]:


set(labels_seen) - set(labels_all)


# In[17]:


set(labels_unseen) - set(labels_all)


# In[ ]:





# In[17]:


class_id_to_split = {}
class_name_to_split = {}
for item in coco_val_anno_all['categories']:
    if item['name'] in labels_seen:
        class_id_to_split[item['id']] = 'seen'
        class_name_to_split[item['name']] = 'seen'
    elif item['name'] in labels_unseen:
        class_id_to_split[item['id']] = 'unseen'
        class_name_to_split[item['name']] = 'unseen'


# In[ ]:





# In[18]:


class_name_to_glove = {}
with open('../datasets/coco/zero-shot/glove.6B.300d.txt', 'r') as fin:
    for row in fin:
        row_tk = row.split()
        if row_tk[0] in class_name_to_split:
            class_name_to_glove[row_tk[0]] = [float(num) for num in row_tk[1:]]


# In[ ]:





# In[19]:


bert = BERT(cfg)


# In[20]:


_ = bert.to('cuda')


# In[21]:


class_name_to_bertemb = {}
for c in class_name_to_split:
    if c not in bert.tokenizer.vocab:
        print(f'{c} not found')
        continue
    cid = bert.tokenizer.vocab[c]
    class_name_to_bertemb[c] = bert.embeddings[cid]


# In[22]:


class_list = list(class_name_to_split.keys())


# In[23]:
import pdb; pdb.set_trace()

encoded_class_list = bert(class_list)


# In[24]:


mask = (1 - encoded_class_list['special_tokens_mask']).to(torch.float32)


# In[25]:


mask.sum(-1)


# In[26]:


embeddings = (encoded_class_list['input_embeddings'] * mask[:, :, None]).sum(1) / mask.sum(1)[:, None]


# In[27]:


embeddings = embeddings.cpu().numpy()


# In[28]:


embeddings.shape


# In[29]:


class_name_to_bertemb = {}
for c, emb in zip(class_list, embeddings.tolist()):
    class_name_to_bertemb[c] = emb


# In[30]:


len(class_name_to_bertemb), len(class_name_to_glove), len(class_name_to_split)


# In[ ]:





# In[31]:


def filter_annotation(anno_dict, split_name_list):
    filtered_categories = []
    for item in anno_dict['categories']:
        if class_id_to_split.get(item['id']) in split_name_list:
            item['embedding'] = {}
            item['embedding']['GloVE'] = class_name_to_glove[item['name']]
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


filter_annotation(coco_train_anno_seen, ['seen'])


# In[33]:


filter_annotation(coco_train_anno_unseen, ['unseen'])


# In[34]:


filter_annotation(coco_train_anno_all, ['seen', 'unseen'])


# In[35]:


filter_annotation(coco_val_anno_seen, ['seen'])


# In[36]:


filter_annotation(coco_val_anno_unseen, ['unseen'])


# In[37]:


filter_annotation(coco_val_anno_all, ['seen', 'unseen'])


# In[38]:


len(coco_val_anno_seen['categories']), len(coco_val_anno_unseen['categories']), len(coco_val_anno_all['categories'])


# In[ ]:





# In[39]:


with open('../datasets/coco/zero-shot/instances_train2017_seen_2.json', 'w') as fout:
    json.dump(coco_train_anno_seen, fout)


# In[40]:


with open('../datasets/coco/zero-shot/instances_train2017_unseen_2.json', 'w') as fout:
    json.dump(coco_train_anno_unseen, fout)


# In[41]:


with open('../datasets/coco/zero-shot/instances_train2017_all_2.json', 'w') as fout:
    json.dump(coco_train_anno_all, fout)


# In[42]:


with open('../datasets/coco/zero-shot/instances_val2017_seen_2.json', 'w') as fout:
    json.dump(coco_val_anno_seen, fout)


# In[43]:


with open('../datasets/coco/zero-shot/instances_val2017_unseen_2.json', 'w') as fout:
    json.dump(coco_val_anno_unseen, fout)


# In[44]:


with open('../datasets/coco/zero-shot/instances_val2017_all_2.json', 'w') as fout:
    json.dump(coco_val_anno_all, fout)


# In[ ]:




