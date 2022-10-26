import random
from typing import List, Tuple
from collections import defaultdict


def train_test_split_int(
        x: List, y: List,
        sample_per_class: int,
        ) -> Tuple[List, List, List, List]:
    """Split dataset by the number of samples in each category.
    
    It is similar to `train_test_split function` of sklearn with `stratify=True`, 
    but the train size is an exact int number. This is designed for few-shot learning,
    to test model performance under small train set.

    Args:
        x: a list of data
        y: label of data, and has the same length with `x`
        sample_per_class: the number of data to be randomly sampled as train set,
        if there are not enough data in some category, this function will divide
        all data of it to train set.

    Returns:
        The same as `train_test_split`, four lists after randomly split.
        They are the lists of train/test data, then the list of train/test label.

    """
    assert len(x) == len(y), 'The number of data and labels must match'

    # the dict to group data by their labels(categories)
    group = defaultdict(list)
    for element, key in zip(x, y):
        group[key].append(element)

    # take out train/test set from each category
    train_x, test_x, train_y, test_y = [], [], [], []
    for key, x_list in group.items():
        random.shuffle(x_list)
        if sample_per_class >= len(x_list):
            train_x.extend(x_list)
            train_y.extend([key] * len(x_list))
        else:
            train_x.extend(x_list[:sample_per_class])
            train_y.extend([key] * sample_per_class)

            test_x.extend(x_list[sample_per_class:])
            test_y.extend([key] * (len(x_list) - sample_per_class))
    return train_x, test_x, train_y, test_y



class AccuracyCounter(object):
    # this is a wrapper of dict, which record the correct number and total
    # occurence of one category

    def __init__(self):
        # Dict{List[correct, total]}
        self.counter = {}

    def __setitem__(self, key, correct):
        if key in self.counter.keys():
            self.counter[key][0] += int(correct)
            self.counter[key][1] += 1
        else:
            self.counter[key] = [int(correct), 1]

    def __getitem__(self, key):
        return self.counter[key]

    def keys(self):
        return self.counter.keys()

    def total_acc(self, method='micro'):
        if method=='micro':
            total = 0
            correct = 0
            for k, (cor, tot) in self.counter.items():
                correct += cor
                total += tot
            return correct / total
        # if method=='marco':
        else:
            acc_sum = 0
            for k, (cor, tot) in self.counter.items():
                acc_sum += cor/tot
            return acc_sum/len(self.counter)

    def acc(self, key):
        cor, tot = self.counter[key]
        return cor/tot

    def analysis(self):
        return {k: v[0] / v[1] for k, v in self.counter.items()}
