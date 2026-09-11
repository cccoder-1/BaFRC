import torch
import torch.utils.data as data
import os
import numpy as np
import random
import json


class FewRelDataset(data.Dataset):
    """
    FewRel Dataset
    """

    def __init__(self, name, pid2name, encoder, N, K, Q, na_rate, root, use_std_desc=True):
        self.root = root
        path = os.path.join(root, name + ".json")
        pid2name_path = os.path.join(root, pid2name + ".json")
        if not os.path.exists(path) or not os.path.exists(pid2name_path):
            print("[ERROR] Data file does not exist!")
            assert 0
        self.json_data = json.load(open(path))
        self.pid2name = json.load(open(pid2name_path))
        self.classes = list(self.json_data.keys())
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        self.NOTA = "none of the above"
        self.NOTA_description = "the relation of the query is not mentioned in the relation set"
        self.encoder = encoder
        self.use_std_desc = bool(use_std_desc)

    def __getraw__(self, item):
        word, pos1, pos2, mask = self.encoder.tokenize(item['tokens'],
                                                       item['h'][2][0],
                                                       item['t'][2][0])
        return word, pos1, pos2, mask

    def __getrel__(self, item):
        word, mask = self.encoder.tokenize_rel(item)
        return word, mask

    def __additem__(self, d, word, pos1, pos2, mask):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)

    def __getitem__(self, index):
        target_classes = random.sample(self.classes, self.N)
        na_classes = list(filter(lambda x: x not in target_classes, self.classes))
        relation_set = {'word': [], 'mask': []}
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        label = []
        query_label = []
        Q_na = int(self.na_rate * self.Q)

        for i, _ in enumerate(target_classes):
            label += [i] * self.K

        for i, class_name in enumerate(target_classes):
            # pid2name/description_pool format:
            #   pid -> [relation_name, orig_desc, std_desc(optional)]
            entry = self.pid2name[class_name]
            rel_name = entry[0] if len(entry) > 0 else ""
            orig_desc = entry[1] if len(entry) > 1 else ""
            std_desc = entry[2] if len(entry) > 2 else ""

            # 原始描述；可选再追加标准化描述（MRM 消融）
            rel_text, rel_text_mask = self.__getrel__([rel_name, orig_desc])
            rel_text, rel_text_mask = torch.tensor(rel_text).long(), torch.tensor(rel_text_mask).long()
            relation_set['word'].append(rel_text)
            relation_set['mask'].append(rel_text_mask)

            if self.use_std_desc:
                rel_text, rel_text_mask = self.__getrel__([rel_name, std_desc])
                rel_text, rel_text_mask = torch.tensor(rel_text).long(), torch.tensor(rel_text_mask).long()
                relation_set['word'].append(rel_text)
                relation_set['mask'].append(rel_text_mask)

            indices = np.random.choice(list(range(len(self.json_data[class_name]))), self.K + self.Q, False)
            count = 0
            for j in indices:
                word, pos1, pos2, mask = self.__getraw__(self.json_data[class_name][j])
                word = torch.tensor(word).long()
                pos1 = torch.tensor(pos1).long()
                pos2 = torch.tensor(pos2).long()
                mask = torch.tensor(mask).long()
                if count < self.K:
                    self.__additem__(support_set, word, pos1, pos2, mask)
                else:
                    self.__additem__(query_set, word, pos1, pos2, mask)
                count += 1
            label += [i] * self.Q
            query_label += [i] * self.Q

        if Q_na > 0:
            # NOTA 描述条数要和上面保持一致：
            # - use_std_desc=True  -> 每类两条（orig/std），NOTA 也两条 => 2N+2
            # - use_std_desc=False -> 每类一条（orig），NOTA 一条 => N+1
            rel_text, rel_text_mask = self.__getrel__([self.NOTA, self.NOTA_description])
            rel_text, rel_text_mask = torch.tensor(rel_text).long(), torch.tensor(rel_text_mask).long()
            relation_set['word'].append(rel_text)
            relation_set['mask'].append(rel_text_mask)
            if self.use_std_desc:
                rel_text, rel_text_mask = self.__getrel__([self.NOTA, self.NOTA_description])
                rel_text, rel_text_mask = torch.tensor(rel_text).long(), torch.tensor(rel_text_mask).long()
                relation_set['word'].append(rel_text)
                relation_set['mask'].append(rel_text_mask)

        # NA
        for j in range(Q_na):
            cur_class = np.random.choice(na_classes, 1, False)[0]
            index = np.random.choice(list(range(len(self.json_data[cur_class]))), 1, False)[0]
            word, pos1, pos2, mask = self.__getraw__(self.json_data[cur_class][index])
            word = torch.tensor(word).long()
            pos1 = torch.tensor(pos1).long()
            pos2 = torch.tensor(pos2).long()
            mask = torch.tensor(mask).long()
            self.__additem__(query_set, word, pos1, pos2, mask)
        query_label += [self.N] * Q_na
        label += [self.N] * Q_na

        return support_set, query_set, query_label, label, relation_set

    def __len__(self):
        return 1000000000


def collate_fn(data):
    batch_support = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_query = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_relation = {'word': [], 'mask': []}
    batch_query_label = []
    batch_label = []
    support_sets, query_sets, query_labels, labels, relation_sets = zip(*data)

    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
        for k in query_sets[i]:
            batch_query[k] += query_sets[i][k]
        for k in relation_sets[i]:
            batch_relation[k] += relation_sets[i][k]
        batch_query_label += query_labels[i]
        batch_label += labels[i]

    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query:
        batch_query[k] = torch.stack(batch_query[k], 0)
    for k in batch_relation:
        batch_relation[k] = torch.stack(batch_relation[k], 0)
    batch_query_label = torch.tensor(batch_query_label)
    batch_label = torch.tensor(batch_label)
    return batch_support, batch_query, batch_query_label, batch_label, batch_relation


def get_loader(name, pid2name, encoder, N, K, Q, na_rate, batch_size, num_workers=0,
               collate_fn=collate_fn, root='./data', use_std_desc=True):
    dataset = FewRelDataset(name, pid2name, encoder, N, K, Q, na_rate, root, use_std_desc=use_std_desc)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    return iter(data_loader)
