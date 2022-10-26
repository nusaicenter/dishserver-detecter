from typing import List
import numpy as np
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.svm import SVC
import pickle
import os
import torch
from torch import nn
import json


class BaseClf(object):
    def __init__(
            self,
            classifier,
            cache_path=None,
            aux_clf_path=None, # path to the CNN last clf layer weights
            aux_id_map=None, # path to the json that mapping clf's id to label
    ):
        if classifier is None:
            raise NotImplementedError

        self.data = []
        self.label = []
        self.num_class = 0
        self.feat_length = None

        self.model = classifier
        self.load_cache(cache_path)

        # use this classifier for extra default prediction
        if aux_clf_path and aux_clf_path:
            # load mobnetv3 last clf layer
            ckpt = torch.load(aux_clf_path, map_location='cpu')
            self.clf = nn.Sequential(
                nn.Linear(960, 1280),
                nn.Hardswish(inplace=True),
                nn.Linear(1280, 1000),
            )
            self.clf.load_state_dict(ckpt)

            # load id-label mapping
            with open(aux_id_map, 'r') as f:
                id2label = json.load(f)
                self.id2label = {int(k): v for k, v in id2label.items()}
        else:
            self.clf = None
            self.id2label = None


    def load_cache(self, cache_path):
        self.cache_path = cache_path or 'clf_model_cache.pkl'
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)

            self.train(cache['data'], cache['label'], all=True)
            print(f'the cache {cache_path} has been loaded to classify model')
        else:
            print(f'the cache of classify model will be save to {cache_path}')

    def clear_cache(self):
        self.data = []
        self.label = []

    def train(self, data: List[np.ndarray], label: List, all: bool=False):
        assert len(data) == len(label), (
            'data and label cannot have different length')

        if all:
            self.data = data
            self.label = label
        else:
            self.data.extend(data)
            self.label.extend(label)

        self.num_class = len(set(self.label))
        if self.num_class > 1:
            self.model.fit(self.data, self.label)
            self.feat_length = len(self.data[0])

        # save all features as cache
        cache = {'data': self.data, 'label': self.label}
        dst_path = self.cache_path.replace('.pkl', '_new.pkl')
        with open(dst_path, 'wb') as f:
            pickle.dump(cache, f)


    def train_all(self, data: List[np.ndarray], label: List):
        self.train(data, label, all=True)


    def train_increment(self, data: List[np.ndarray], label: List):
        self.train(data, label, all=False)


    def predict(self, feats: List[np.ndarray]):
        if self.num_class == 0:
            if self.clf:
                return self.aux_predict(feats)
            else:
                return ['菜品'] * len(feats)
        elif self.num_class == 1:
            return [self.label[0]] * len(feats)
        else:
            return self.model.predict(feats)

    def aux_predict(self, feats):
        feats = torch.FloatTensor(feats) # convert list of ndarray to tensor
        outputs = self.clf(feats) # inference
        # get predict indexes and convert ids to dish names
        preds = outputs.argmax(dim=-1) 
        preds = [self.id2label[p.item()] for p in preds] 
        return preds

    def det_predict(self, feats: List[List[np.ndarray]]):
        indexes = []
        inputs = []
        # flatten nested image feats to 1-d list
        for i, img_feats in enumerate(feats):
            for feat in img_feats:
                inputs.append(feat)
                indexes.append(i)

        # batch inference
        outs = self.predict(inputs) if len(inputs) > 0 else []

        # revert to nested list
        outputs = [[]] * len(feats)
        for idx, out in zip(indexes, outs):
            outputs[idx].append(out)
        return outputs

    def get_label(self):
        return list(set(self.label))


class nearCenter(BaseClf):
    def __init__(self, cache_path=None, aux_clf_path=None, aux_id_map=None):
        super().__init__(classifier=NearestCentroid(),
                         cache_path=cache_path,
                         aux_clf_path=aux_clf_path,
                         aux_id_map=aux_id_map)


class nearNeighbor(BaseClf):
    def __init__(self, cache_path=None, aux_clf_path=None, aux_id_map=None):
        super().__init__(classifier=KNeighborsClassifier(),
                         cache_path=cache_path,
                         aux_clf_path=aux_clf_path,
                         aux_id_map=aux_id_map)

class svm(BaseClf):
    def __init__(self, cache_path=None, aux_clf_path=None, aux_id_map=None):
        super().__init__(classifier=SVC(),
                         cache_path=cache_path,
                         aux_clf_path=aux_clf_path,
                         aux_id_map=aux_id_map)