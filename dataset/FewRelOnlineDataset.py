import torch
import torch.utils.data as data
import os, json


class FewRelOnlineDataset(data.Dataset):
    """
    Codalab FewRel NOTA test input:
    each episode has:
      - meta_train: [N][K] samples
      - meta_test:  one sample
      - relation:   [N] pid list aligned with meta_train order (IMPORTANT)
    """

    def __init__(self, name, pid2name, encoder, N, K, root):
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"[ERROR] Online test file does not exist: {path}")
        self.episodes = json.load(open(path, "r", encoding="utf-8"))
        self.encoder = encoder
        self.N = N
        self.K = K

        pid2name_path = os.path.join(root, pid2name + ".json")
        if not os.path.exists(pid2name_path):
            raise FileNotFoundError(f"[ERROR] pid2name file does not exist: {pid2name_path}")
        self.pid2name = json.load(open(pid2name_path, "r", encoding="utf-8"))

        self.NOTA = "none of the above"
        self.NOTA_description = "the relation of the query is not mentioned in the relation set"

    def __getraw__(self, item):
        word, pos1, pos2, mask = self.encoder.tokenize(
            item['tokens'], item['h'][2][0], item['t'][2][0]
        )
        return (torch.tensor(word).long(),
                torch.tensor(pos1).long(),
                torch.tensor(pos2).long(),
                torch.tensor(mask).long())

    def __getrel__(self, rel_pair):
        word, mask = self.encoder.tokenize_rel(rel_pair)
        return torch.tensor(word).long(), torch.tensor(mask).long()

    def __getitem__(self, idx):
        ep = self.episodes[idx]
        meta_train = ep["meta_train"]   # [N][K]
        meta_test  = ep["meta_test"]    # 1 dict

        # IMPORTANT: pid list aligned with meta_train order
        pid_list = ep.get("relation", None)
        if pid_list is None:
            raise KeyError("[ERROR] Online episode has no 'relation' pid list.")
        if len(pid_list) != self.N:
            raise ValueError(f"[ERROR] len(relation)={len(pid_list)} != N={self.N}")

        support = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        rel_text = {'word': [], 'mask': []}

        # build support and per-class relation text
        for i in range(self.N):
            cls_samples = meta_train[i]
            if len(cls_samples) != self.K:
                raise ValueError(f"[ERROR] meta_train[{i}] has {len(cls_samples)} != K={self.K}")

            for j in range(self.K):
                w, p1, p2, m = self.__getraw__(cls_samples[j])
                support['word'].append(w)
                support['pos1'].append(p1)
                support['pos2'].append(p2)
                support['mask'].append(m)

            pid = pid_list[i]
            if pid not in self.pid2name:
                # Fall back to empty descriptions when pid2name lacks this id.
                rel_name, orig_desc, std_desc = "", "", ""
            else:
                entry = self.pid2name[pid]  # [name, orig_desc, std_desc(optional)]
                rel_name = entry[0] if len(entry) > 0 else ""
                orig_desc = entry[1] if len(entry) > 1 else ""
                std_desc = entry[2] if len(entry) > 2 else ""

            # Add TWO relation descriptions per class: orig then std.
            rw, rm = self.__getrel__([rel_name, orig_desc])
            rel_text['word'].append(rw)
            rel_text['mask'].append(rm)
            rw, rm = self.__getrel__([rel_name, std_desc])
            rel_text['word'].append(rw)
            rel_text['mask'].append(rm)

        # append NOTA twice to keep (2N+2) relation texts
        rw, rm = self.__getrel__([self.NOTA, self.NOTA_description])
        rel_text['word'].append(rw)
        rel_text['mask'].append(rm)
        rw, rm = self.__getrel__([self.NOTA, self.NOTA_description])
        rel_text['word'].append(rw)
        rel_text['mask'].append(rm)

        # query
        query = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        w, p1, p2, m = self.__getraw__(meta_test)
        query['word'].append(w)
        query['pos1'].append(p1)
        query['pos2'].append(p2)
        query['mask'].append(m)

        # stack episode tensors
        for k in support:
            support[k] = torch.stack(support[k], 0)   # (N*K, L)
        for k in query:
            query[k] = torch.stack(query[k], 0)       # (1, L)
        for k in rel_text:
            rel_text[k] = torch.stack(rel_text[k], 0) # (N+1, L_rel)

        return support, query, rel_text

    def __len__(self):
        return len(self.episodes)


def collate_fn_online(batch):
    batch_support = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_query   = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_rel     = {'word': [], 'mask': []}

    supports, queries, rels = zip(*batch)
    for i in range(len(supports)):
        for k in batch_support:
            batch_support[k].append(supports[i][k])
        for k in batch_query:
            batch_query[k].append(queries[i][k])
        for k in batch_rel:
            batch_rel[k].append(rels[i][k])

    for k in batch_support:
        batch_support[k] = torch.cat(batch_support[k], 0)  # (B*N*K, L)
    for k in batch_query:
        batch_query[k] = torch.cat(batch_query[k], 0)      # (B*1, L)
    for k in batch_rel:
        batch_rel[k] = torch.cat(batch_rel[k], 0)          # (B*(N+1), L_rel)

    return batch_support, batch_query, batch_rel


def get_loader_online(name, pid2name, encoder, N, K, batch_size, num_workers=0, root='./data'):
    dataset = FewRelOnlineDataset(name, pid2name, encoder, N, K, root)
    loader = data.DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=num_workers,
                             collate_fn=collate_fn_online)
    return iter(loader), len(dataset)
