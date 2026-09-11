import json
import os
import torch
import torch.utils.data as data


class FSTacredEpisodeDataset(data.Dataset):
    """
    Load pre-sampled FS-TACRED episodes.
    Expected file format:
      top-level list with at least one element;
      first element is a list of episodes, each with:
        - meta_train: List[List[sample]] of shape (N, K)
        - meta_test: List[sample] (often length 1)

    Optional relation text: ``{root}/{pid2name}.json`` (same as FewRel / description_pool).
    Each relation id maps to [relation_name, orig_desc, std_desc]; if missing, synthetic
    TACRED templates are used.
    """

    def __init__(self, name, pid2name, encoder, N, K, Q, na_rate, root, use_std_desc=True):
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"[ERROR] FS-TACRED file does not exist: {path}")

        raw = json.load(open(path, "r", encoding="utf-8"))
        if not isinstance(raw, list) or len(raw) < 1 or not isinstance(raw[0], list):
            raise ValueError("[ERROR] Unexpected FS-TACRED JSON format.")

        pid2name_path = os.path.join(root, pid2name + ".json")
        if os.path.isfile(pid2name_path):
            self.pid2name = json.load(open(pid2name_path, "r", encoding="utf-8"))
        else:
            self.pid2name = None
            print(
                f"[FSTacredEpisodeDataset] No {pid2name}.json under {root}; "
                "using synthetic TACRED relation strings."
            )

        self.episodes = raw[0]
        self.encoder = encoder
        self.N = int(N)
        self.K = int(K)
        self.Q = int(Q)
        self.na_rate = int(na_rate)
        self.use_std_desc = bool(use_std_desc)
        self.NOTA = "none of the above"
        self.NOTA_DESCRIPTION = "the relation of the query is not mentioned in the relation set"

    def __len__(self):
        return len(self.episodes)

    def __getraw__(self, item):
        word, pos1, pos2, mask = self.encoder.tokenize(
            item["tokens"], item["h"][2][0], item["t"][2][0]
        )
        return (
            torch.tensor(word).long(),
            torch.tensor(pos1).long(),
            torch.tensor(pos2).long(),
            torch.tensor(mask).long(),
        )

    def __getrel__(self, rel_pair):
        word, mask = self.encoder.tokenize_rel(rel_pair)
        return torch.tensor(word).long(), torch.tensor(mask).long()

    def _rel_text_pair(self, rel_name):
        # Align with FewRelDataset: pid -> [rel_name, orig_desc, std_desc]
        if self.pid2name is not None and rel_name in self.pid2name:
            entry = self.pid2name[rel_name]
            rel_name_txt = entry[0] if len(entry) > 0 else str(rel_name).replace("_", " ")
            orig_desc = entry[1] if len(entry) > 1 else f"TACRED relation {rel_name}"
            std_desc = entry[2] if len(entry) > 2 else ""
            return [rel_name_txt, orig_desc, std_desc]
        short = str(rel_name).replace("_", " ")
        orig_desc = f"TACRED relation {rel_name}"
        std_desc = f"The head entity and tail entity satisfy relation: {short}."
        return [short, orig_desc, std_desc]

    def __additem__(self, d, word, pos1, pos2, mask):
        d["word"].append(word)
        d["pos1"].append(pos1)
        d["pos2"].append(pos2)
        d["mask"].append(mask)

    def __getitem__(self, index):
        ep = self.episodes[index]
        meta_train = ep["meta_train"]  # (N, K)
        meta_test = ep["meta_test"]    # list of queries

        support_set = {"word": [], "pos1": [], "pos2": [], "mask": []}
        query_set = {"word": [], "pos1": [], "pos2": [], "mask": []}
        relation_set = {"word": [], "mask": []}
        label = []
        query_label = []

        if len(meta_train) != self.N:
            raise ValueError(f"[ERROR] Episode N mismatch: got {len(meta_train)} vs args N={self.N}")

        class_rel = []
        for i in range(self.N):
            cls_samples = meta_train[i]
            if len(cls_samples) != self.K:
                raise ValueError(f"[ERROR] Episode K mismatch at class {i}: got {len(cls_samples)} vs args K={self.K}")

            rel_name = cls_samples[0].get("relation", f"class_{i}")
            class_rel.append(rel_name)

            rel_name_txt, rel_desc, rel_std = self._rel_text_pair(rel_name)
            rw, rm = self.__getrel__([rel_name_txt, rel_desc])
            relation_set["word"].append(rw)
            relation_set["mask"].append(rm)
            if self.use_std_desc:
                rw, rm = self.__getrel__([rel_name_txt, rel_std])
                relation_set["word"].append(rw)
                relation_set["mask"].append(rm)

            for sample in cls_samples:
                w, p1, p2, m = self.__getraw__(sample)
                self.__additem__(support_set, w, p1, p2, m)
            label += [i] * self.K

        # Keep relation_set length behavior aligned with FewRelDataset.
        if self.na_rate > 0:
            rw, rm = self.__getrel__([self.NOTA, self.NOTA_DESCRIPTION])
            relation_set["word"].append(rw)
            relation_set["mask"].append(rm)
            if self.use_std_desc:
                rw, rm = self.__getrel__([self.NOTA, self.NOTA_DESCRIPTION])
                relation_set["word"].append(rw)
                relation_set["mask"].append(rm)

        # Use all meta_test queries from the pre-sampled episode.
        for sample in meta_test:
            w, p1, p2, m = self.__getraw__(sample)
            self.__additem__(query_set, w, p1, p2, m)
            q_rel = sample.get("relation", "no_relation")
            if q_rel in class_rel:
                y = class_rel.index(q_rel)
            else:
                y = self.N
            query_label.append(y)
            label.append(y)

        return support_set, query_set, query_label, label, relation_set


def collate_fn_fs_tacred(data_batch):
    batch_support = {"word": [], "pos1": [], "pos2": [], "mask": []}
    batch_query = {"word": [], "pos1": [], "pos2": [], "mask": []}
    batch_relation = {"word": [], "mask": []}
    batch_query_label = []
    batch_label = []

    support_sets, query_sets, query_labels, labels, relation_sets = zip(*data_batch)
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


def get_loader_fs_tacred(name, pid2name, encoder, N, K, Q, na_rate, batch_size, num_workers=0,
                         collate_fn=collate_fn_fs_tacred, root="./FS-TACRED", use_std_desc=True):
    dataset = FSTacredEpisodeDataset(
        name=name, pid2name=pid2name, encoder=encoder, N=N, K=K, Q=Q, na_rate=na_rate,
        root=root, use_std_desc=use_std_desc
    )
    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    def _infinite_iter(loader):
        it = iter(loader)
        while True:
            try:
                yield next(it)
            except StopIteration:
                it = iter(loader)

    return iter(_infinite_iter(data_loader))
