from typing import List
import numpy as np
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.svm import SVC
import pickle
import os


class BaseClf(object):
    def __init__(self, classifier, cache_path=None):
        if classifier is None:
            raise NotImplementedError

        self.data = []
        self.label = []
        self.num_class = 0
        self.feat_length = None

        self.model = classifier
        self.load_cache(cache_path)


    def load_cache(self, cache_path):
        self.cache_path = cache_path or 'clf_model_cache.pkl'
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                cache = pickle.load(f)

            self.train(cache['data'], cache['label'], all=True)
            print(f'the cache {self.cache_path} has been loaded to classify model')
        else:
            print(f'the cache of classify model will be save to {self.cache_path}')


    def train(self, data: List[np.ndarray], label: List, all: bool):
        assert len(data) > 0 and len(data) == len(label), (
            'data and label cannot be empty or have the different length')

        if all:
            self.data = data
            self.label = label
        else:
            self.data.extend(data)
            self.label.extend(label)

        self.model.fit(self.data, self.label)
        self.num_class = len(set(self.label))
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
            return [-1] * len(feats)
        else:
            return self.model.predict(feats)


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
    def __init__(self, cache_path=None):
        super().__init__(classifier=NearestCentroid(), cache_path=cache_path)


class nearNeighbor(BaseClf):
    def __init__(self, cache_path=None):
        super().__init__(classifier=KNeighborsClassifier(), cache_path=cache_path)


class svm(BaseClf):
    def __init__(self, cache_path=None):
        super().__init__(classifier=SVC(), cache_path=cache_path)
