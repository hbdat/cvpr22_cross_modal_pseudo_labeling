
from .lvis_v1_categories import LVIS_CATEGORIES

from tqdm import tqdm
import spacy
import pickle
from nltk.corpus import wordnet as wn


def normalize_class_names(thing_classes):
    new_thing_classes = []
    for name in thing_classes:
        new_name = name.replace('_',' ')
        new_name = new_name.replace('/',' ')
        new_name = new_name.replace('(',' ')
        new_name = new_name.replace(')',' ')
        new_name = new_name.lower()

        new_thing_classes.append(new_name)

    return new_thing_classes

class LVISParser():
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.look_up = {}
        self.class_names = ['']*len(LVIS_CATEGORIES)
        for item in LVIS_CATEGORIES:
            synonyms = item['synonyms']

            synonyms = [s.lower() for s in synonyms]
            synonyms = [s.replace('_',' ') for s in synonyms]

            id = item['id']-1       # convert to 0 base

            self.class_names[id] = item['name']

            for s in synonyms:
                doc = self.nlp(s)

                lemma_s = []
                for token in doc:
                    word = token.lemma_
                    if word.startswith('('):    #<< skip word in ()
                        break
                    lemma_s.append(word)
                lemma_s = ' '.join(lemma_s)
                lemma_s = lemma_s.replace(' - ','-')

                # if lemma_s in self.look_up:
                #     print('Duplication {}'.format(lemma_s))

                self.look_up[lemma_s] = id

        print('lvis parser vocab size {}'.format(len(self.look_up)))

    def parse(self, sentence):
        sentence = sentence.lower()

        doc = self.nlp(sentence)
        lemma_sentence = []
        for token in doc:
            lemma_sentence.append(token.lemma_)
        lemma_sentence = ' '.join(lemma_sentence)

        nns = []
        category_ids = []

        for s in self.look_up:
            if ' {} '.format(s) in lemma_sentence or lemma_sentence.startswith(s+' ') or lemma_sentence.endswith(' '+s) or lemma_sentence == s:
                nns.append(s)
                category_ids.append(self.look_up[s])
        
        return nns, category_ids

# class WordNetParser():
#     def __init__(self):
#         self.nlp = spacy.load("en_core_web_sm")
#         self.look_up = {}
    
#     def is_object(word, category):       ## is_a relationship
#         # Assume the category always uses the most popular sense
#         cat_syn = wn.synsets(category)[0]

#         word_senses = wn.synsets(word)
#         if len(word_senses) == 0:
#             print('cannot find {} !!!'.format(word))
#         # For the input, check all senses
#         for syn in word_senses:
#             for match in syn.lowest_common_hypernyms(cat_syn):
#                 if match == cat_syn:
#                     return True
#         return False

#     def parse(self, sentence):
#         sentence = sentence.lower()

#         doc = self.nlp(sentence)
#         lemma_sentence = []
#         for token in doc:
#             lemma_sentence.append(token.lemma_)
#         lemma_sentence = ' '.join(lemma_sentence)

#         nns = []
#         category_ids = []

#         for s in self.look_up:
#             if ' {} '.format(s) in lemma_sentence or lemma_sentence.startswith(s+' ') or lemma_sentence.endswith(' '+s) or lemma_sentence == s:
#                 nns.append(s)
#                 category_ids.append(self.look_up[s])
        
#         return nns, category_ids