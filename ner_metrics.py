import torch
from collections import Counter
from utils_ner import get_entities

# class SeqEntityScore(object):
#     def __init__(self, id2label,markup='bios'):
#         self.id2label = id2label
#         self.markup = markup
#         self.reset()
#
#     def reset(self):
#         self.origins = []
#         self.founds = []
#         self.rights = []
#
#     def compute(self, origin, found, right):
#         """
#         Args:
#             origin: 原本的标签
#             found: 预测出来的标签
#             right: 预测标签中正确的标签
#
#         Returns:计算出来的精确率 召回率  f1值
#
#         """
#         recall = 0 if origin == 0 else (right / origin)
#         precision = 0 if found == 0 else (right / found)
#         f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
#         return recall, precision, f1
#
#     def result(self):
#         """
#         根据 origin: 原本的标签
#             found: 预测出来的标签
#             right: 预测标签中正确的标签
#
#             直接可算总的recall, precision, f1
#
#             class_info中保存各个标签类型的recall, precision, f1
#         """
#
#         class_info = {}
#         origin_counter = Counter([x[0] for x in self.origins]) #x[0]:标签
#         found_counter = Counter([x[0] for x in self.founds])
#         right_counter = Counter([x[0] for x in self.rights])
#         for type_, count in origin_counter.items():
#             origin = count
#             found = found_counter.get(type_, 0)
#             right = right_counter.get(type_, 0)
#             recall, precision, f1 = self.compute(origin, found, right)
#             class_info[type_] = {"acc": round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4)}
#
#         origin = len(self.origins) #3072
#         found = len(self.founds)   #2814
#         right = len(self.rights)   #1184
#         recall, precision, f1 = self.compute(origin, found, right)
#         return {'acc': precision, 'recall': recall, 'f1': f1}, class_info
#
#     def update(self, label_paths, pred_paths):
#         '''
#         labels_paths: [[],[],[],....]
#         pred_paths: [[],[],[],.....]
#
#         :param label_paths:
#         :param pred_paths:
#         :return:
#         Example:
#             >>> labels_paths = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
#             >>> pred_paths = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
#         '''
#         for label_path, pre_path in zip(label_paths, pred_paths):
#             label_entities = get_entities(label_path, self.id2label,self.markup) #[['name', 16, 19]]
#             pre_entities = get_entities(pre_path, self.id2label,self.markup)     #[['organization', 16, 20]]
#             self.origins.extend(label_entities)
#             self.founds.extend(pre_entities)
#             self.rights.extend([pre_entity for pre_entity in pre_entities if pre_entity in label_entities])


import numpy as np
#colab数据集
class SeqEntityScore(object):
    def __init__(self, id2label,markup='bieos'):
        self.id2label = id2label
        self.markup = markup
        self.reset()

    def reset(self):
        self.origins = []
        self.founds = []
        self.rights = []

    def compute(self, origin, found, right):
        """
        Args:
            origin: 原本的标签
            found: 预测出来的标签
            right: 预测标签中正确的标签

        Returns:计算出来的精确率 召回率  f1值

        """
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def result(self):
        """
        根据 origin: 原本的标签
            found: 预测出来的标签       (a,(0,3))
            right: 预测标签中正确的标签
            直接算总的recall, precision, f1

        """

        origin = len(self.origins)
        found = len(self.founds)
        right = len(self.rights)
        recall, precision, f1 = self.compute(origin, found, right)

        return f1

    def update(self, label_paths, pred_paths):
        '''
        labels_paths: [[],[],[],....]
        pred_paths: [[],[],[],.....]

        :param label_paths:
        :param pred_paths:
        :return:
        Example:
            >>> labels_paths = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
            >>> pred_paths = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        '''

        for label_path, pre_path in zip(label_paths, pred_paths):
            label_entities = get_entities(label_path,self.id2label)
            pre_entities = get_entities(pre_path,self.id2label)
            self.origins.extend(label_entities)
            self.founds.extend(pre_entities)
            self.rights.extend([pre_entity for pre_entity in pre_entities if pre_entity in label_entities])

    def compute_train_pre(self, label_paths, pred_paths):
        """
        train过程中计算每个batch的精确率
        """
        #origin = []
        found = []
        right = []
        for label_path, pre_path in zip(label_paths, pred_paths):
            label_entities = get_entities(label_path,self.id2label)
            pre_entities = get_entities(pre_path,self.id2label)
            #origin.extend(label_entities)
            found.extend(pre_entities)
            right.extend([pre_entity for pre_entity in pre_entities if pre_entity in label_entities])

        return 0 if len(found) == 0 else len(right) / len(found)




class SpanEntityScore(object):
    def __init__(self, id2label):
        self.id2label = id2label
        self.reset()

    def reset(self):
        self.origins = []
        self.founds = []
        self.rights = []

    def compute(self, origin, found, right):
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def result(self):
        class_info = {}
        origin_counter = Counter([self.id2label[x[0]] for x in self.origins])
        found_counter = Counter([self.id2label[x[0]] for x in self.founds])
        right_counter = Counter([self.id2label[x[0]] for x in self.rights])
        for type_, count in origin_counter.items():
            origin = count
            found = found_counter.get(type_, 0)
            right = right_counter.get(type_, 0)
            recall, precision, f1 = self.compute(origin, found, right)
            class_info[type_] = {"acc": round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4)}
        origin = len(self.origins)
        found = len(self.founds)
        right = len(self.rights)
        recall, precision, f1 = self.compute(origin, found, right)
        return {'acc': precision, 'recall': recall, 'f1': f1}, class_info

    def update(self, true_subject, pred_subject):
        self.origins.extend(true_subject)
        self.founds.extend(pred_subject)
        self.rights.extend([pre_entity for pre_entity in pred_subject if pre_entity in true_subject])


if __name__ == '__main__':
    id2label = []
    test = SeqEntityScore(id2label=id2label)

    label_paths = [['B-a','I-a','E-a','O','S-fac'],['B-a','E-a','S-a','O','S-e']]
    pred_paths = [['B-a','B-a','I-a','E-a','S-fac'],['B-a','I-a','S-e','O','O']]
    res =test.compute_train_pre(label_paths,pred_paths)
    print(res)