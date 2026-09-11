import os
import torch
import numpy as np
import json, time
from torch import optim, nn
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from tqdm import tqdm


def _print_if(condition, *args, **kwargs):
    if condition:
        print(*args, **kwargs)




class FewShotREModel(nn.Module):
    def __init__(self, my_sentence_encoder):
        """
        sentence_encoder: Sentence encoder

        You need to set self.cost as your own loss function.
        """
        nn.Module.__init__(self)
        self.sentence_encoder = my_sentence_encoder  # nn.DataParallel(my_sentence_encoder)
        self.cost = nn.CrossEntropyLoss()

    def forward(self, support, query, relation, N, K, total_Q):
        """
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        total_Q: Num of instances for each class in the query set
        return: logits, pred
        """
        raise NotImplementedError

    def l2norm(self, X):
        norm = torch.pow(X, 2).sum(dim=-1, keepdim=True).sqrt()
        X = torch.div(X, norm)
        return X

    def loss(self, logits, label):
        """
        logits: Logits with the size (..., class_num)
        label: Label with whatever size.
        return: [Loss] (A single value)
        """
        N = logits.size(-1)
        return self.cost(logits.view(-1, N), label.view(-1))

    def accuracy(self, pred, label):
        """
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        """
        return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))


class FewShotREFramework:

    def __init__(self, train_data_loader, val_data_loader):
        """
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        """
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        

    def __load_model__(self, ckpt):
        """
        ckpt: Path of the checkpoint
        return: Checkpoint dict
        """
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            print("Successfully loaded checkpoint '%s'" % ckpt)
            return checkpoint
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)

    def item(self, x):
        """
        PyTorch before and after 0.4
        """
        torch_version = torch.__version__.split('.')
        if int(torch_version[0]) == 0 and int(torch_version[1]) < 4:
            return x[0]
        else:
            return x.item()

    @staticmethod
    def _compute_target_micro_metrics(tp, fp, fn):
        precision = tp / max(tp + fp, 1e-12)
        recall = tp / max(tp + fn, 1e-12)
        f1 = 0.0 if (precision + recall) < 1e-12 else (2 * precision * recall / (precision + recall))
        return precision, recall, f1

    def train(self,
              model,
              model_name,
              B, N, K, Q,
              na_rate,
              learning_rate=2e-5,
              weight_decay=1e-5,
              train_iter=30000,
              val_iter=1000,
              val_step=1000,
              load_ckpt=None,
              save_ckpt=None,
              warmup_step=300,
              grad_iter=1,
              primary_metric='legacy',
              early_stopping_patience=6):
        """
        model: Few-shot RE model
        model_name: Name of the model
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        na_rate: NOTA rate in training
        learning_rate: Initial learning rate
        weight_decay: Rate of decaying weight
        train_iter: Num of iterations of training
        val_iter: Num of iterations of validating
        val_step: Validate every val_step steps
        """
        print("Start training...")

        # Init
        parameters_to_optimize = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        parameters_to_optimize = [
            {'params': [p for n, p in parameters_to_optimize
                        if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in parameters_to_optimize
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(parameters_to_optimize, lr=learning_rate, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step,
                                                    num_training_steps=train_iter)

        if load_ckpt:
            state_dict = self.__load_model__(load_ckpt)['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    continue
                own_state[name].copy_(param)
            start_iter = 0
        else:
            start_iter = 0

        model.train()

        # Training
        best_metric = -1e18
        iter_loss = 0.0
        iter_right = 0.0
        iter_f1_macro = 0.0
        iter_sample = 0.0
        # extra metrics
        nota_tp = 0.0
        nota_fp = 0.0
        nota_fn = 0.0
        known_right = 0.0
        known_total = 0.0
        target_tp = 0.0
        target_fp = 0.0
        target_fn = 0.0
        gold_known_count = 0.0
        pred_known_count = 0.0
        total_count = 0.0
        early_stopping_step = 0

        if primary_metric == 'target_micro_f1':
            print("[TRAIN] best-checkpoint metric: target_micro_f1")
        else:
            print("[TRAIN] best-checkpoint metric: acc + macro_f1 (legacy)")

        for it in range(start_iter, start_iter + train_iter):
            support, query, query_label, label, rel_text = next(self.train_data_loader)
            if torch.cuda.is_available():
                for k in support:
                    support[k] = support[k].cuda()
                for k in query:
                    query[k] = query[k].cuda()
                for k in rel_text:
                    rel_text[k] = rel_text[k].cuda()
                query_label = query_label.cuda()
            # For pre-sampled episodic datasets (e.g., FS-TACRED), total query
            # count may not equal N*Q+Q*na_rate. Infer it from the actual batch.
            total_Q = query_label.size(0) // B
            query_label = query_label.view(B, total_Q)

            # BaFRC forward convention: (logits, pred, aux)
            logits, pred, _ = model(support, query, rel_text, N, K, Q, total_Q)
            loss = model.loss(logits, None, query_label, N, Q) / float(grad_iter)

            right = model.accuracy(pred, query_label)
            f1_macro = f1_score(y_true=query_label.view(-1).cpu().numpy(), y_pred=pred.cpu().view(-1).numpy(),
                                average='macro')

            # ---- NOTA (as positive) precision/recall/f1 and known-only accuracy ----
            y_true = query_label.view(-1)
            y_pred = pred.view(-1)
            nota_true = (y_true == N)
            nota_pred = (y_pred == N)
            nota_tp += float((nota_true & nota_pred).sum().item())
            nota_fp += float((~nota_true & nota_pred).sum().item())
            nota_fn += float((nota_true & ~nota_pred).sum().item())

            known_mask = (y_true < N)
            known_total += float(known_mask.sum().item())
            if known_mask.any():
                known_right += float((y_pred[known_mask] == y_true[known_mask]).sum().item())

            target_tp += float(((y_true < N) & (y_pred < N) & (y_true == y_pred)).sum().item())
            target_fp += float((((y_true == N) & (y_pred < N)) | ((y_true < N) & (y_pred < N) & (y_true != y_pred))).sum().item())
            target_fn += float((((y_true < N) & (y_pred == N)) | ((y_true < N) & (y_pred < N) & (y_true != y_pred))).sum().item())
            gold_known_count += float((y_true < N).sum().item())
            pred_known_count += float((y_pred < N).sum().item())
            total_count += float(y_true.numel())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

            if it % grad_iter == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            iter_loss += self.item(loss.data)
            iter_right += self.item(right.data)
            iter_f1_macro += f1_macro
            iter_sample += 1

            nota_prec = nota_tp / max(nota_tp + nota_fp, 1e-12)
            nota_rec = nota_tp / max(nota_tp + nota_fn, 1e-12)
            nota_f1 = 0.0 if (nota_prec + nota_rec) < 1e-12 else (2 * nota_prec * nota_rec / (nota_prec + nota_rec))
            known_acc = known_right / max(known_total, 1e-12)
            target_p, target_r, target_f1 = self._compute_target_micro_metrics(target_tp, target_fp, target_fn)
            gold_known_ratio = gold_known_count / max(total_count, 1e-12)
            pred_known_ratio = pred_known_count / max(total_count, 1e-12)

            _print_if(
                (it + 1) % min(100, val_step) == 0,
                'step: {0:4} | loss: {1:2.6f} | acc: {2:3.2f}%, macro_f1: {3:3.2f}% | '
                'known_acc: {4:3.2f}% | NOTA P/R/F1: {5:3.2f}/{6:3.2f}/{7:3.2f}% | '
                'target_micro P/R/F1: {8:3.2f}/{9:3.2f}/{10:3.2f}% | '
                'gold/pred_known: {11:3.2f}/{12:3.2f}%'.format(
                    it + 1,
                    iter_loss / iter_sample,
                    100 * iter_right / iter_sample,
                    100 * iter_f1_macro / iter_sample,
                    100 * known_acc,
                    100 * nota_prec,
                    100 * nota_rec,
                    100 * nota_f1,
                    100 * target_p,
                    100 * target_r,
                    100 * target_f1,
                    100 * gold_known_ratio,
                    100 * pred_known_ratio,
                ),
                end='\r',
                flush=True,
            )

            

            if (it + 1) % val_step == 0:

                acc, f1, _, _, _, _, _, _, target_f1_val, _, _ = self.eval(
                    model, model_name, B, N, K, Q, na_rate, val_iter
                )
                model.train()
                if primary_metric == 'target_micro_f1':
                    score = target_f1_val
                else:
                    score = acc + f1
                if score > best_metric:
                    print('Best checkpoint')
                    torch.save({'state_dict': model.state_dict()}, save_ckpt)
                    best_metric = score
                    early_stopping_step = 0
                else:
                    early_stopping_step += 1
                    if (
                        early_stopping_patience > 0
                        and early_stopping_step >= early_stopping_patience
                    ):
                        print("early stopping...")
                        break

                iter_loss = 0.0
                iter_right = 0.0
                iter_f1_macro = 0.0
                iter_sample = 0.0
                nota_tp = 0.0
                nota_fp = 0.0
                nota_fn = 0.0
                known_right = 0.0
                known_total = 0.0
                target_tp = 0.0
                target_fp = 0.0
                target_fn = 0.0
                gold_known_count = 0.0
                pred_known_count = 0.0
                total_count = 0.0

        print(f"Training finished: {model_name}")

    def eval(self,
             model,
             model_name,
             B, N, K, Q,
             na_rate,
             eval_iter,
             ckpt=None,
             test_data_loader=None,
             use_joint_reject=False,
             joint_lambda_pref=0.5,
             joint_tau_reject=0.0):
        model.eval()
        if ckpt is None:
            eval_dataset = self.val_data_loader
        else:
            if ckpt != 'none':
                state_dict = self.__load_model__(ckpt)['state_dict']
                own_state = model.state_dict()
                for name, param in state_dict.items():
                    if name not in own_state:
                        continue
                    own_state[name].copy_(param)
            eval_dataset = test_data_loader

        iter_right = 0.0
        iter_f1_macro = 0.0
        iter_sample = 0.0
        # extra metrics
        nota_tp = 0.0
        nota_fp = 0.0
        nota_fn = 0.0
        known_right = 0.0
        known_total = 0.0
        target_tp = 0.0
        target_fp = 0.0
        target_fn = 0.0
        gold_known_count = 0.0
        pred_known_count = 0.0
        total_count = 0.0
        with torch.no_grad():
            for it in range(eval_iter):
                support, query, query_label, label, rel_text = next(eval_dataset)
                if torch.cuda.is_available():
                    for k in support:
                        support[k] = support[k].cuda()
                    for k in query:
                        query[k] = query[k].cuda()
                    for k in rel_text:
                        rel_text[k] = rel_text[k].cuda()
                    query_label = query_label.cuda()
        
                # ---- 关键修改：按实际 batch 反推每个 episode 的 query 数 ----
                # query_label 现在是一维：长度 = B * total_Q
                total_Q = query_label.size(0) // B           # 每个 episode 的总 query 数（含 NA）
                query_label = query_label.view(B, total_Q)   # (B, total_Q)
        
                logits, pred, _ = model(support, query, rel_text, N, K, Q, total_Q)

                # Optional offline joint reject at eval time only:
                # score_joint = (d1 - R_nearest) - lambda_pref * (d2 - d1)
                # predict NOTA iff score_joint > tau_reject
                pred_eval = pred
                cache = getattr(model, "_cache", {})
                dists_q = cache.get("dists_q", None)      # (B,total_Q,N)
                R_sup = cache.get("R_support", None)      # (B,N)
                if use_joint_reject and dists_q is not None and R_sup is not None:
                    top2 = torch.topk(dists_q, k=min(2, N), dim=-1, largest=False).values
                    d1 = top2[..., 0]
                    d2 = top2[..., 1] if N >= 2 else d1
                    nearest = dists_q.argmin(dim=-1)
                    R_nearest = R_sup.gather(1, nearest)
                    g = d1 - R_nearest
                    pref = d2 - d1
                    score_joint = g - float(joint_lambda_pref) * pref
                    pred_eval = nearest.clone()
                    pred_eval[score_joint > float(joint_tau_reject)] = N

                right = model.accuracy(pred_eval, query_label)
                f1_macro = f1_score(
                    y_true=query_label.view(-1).cpu().numpy(),
                    y_pred=pred_eval.cpu().view(-1).numpy(),
                    average='macro'
                )

                # ---- NOTA (as positive) precision/recall/f1 and known-only accuracy ----
                y_true = query_label.view(-1)
                y_pred = pred_eval.view(-1)

                nota_true = (y_true == N)
                nota_pred = (y_pred == N)
                nota_tp += float((nota_true & nota_pred).sum().item())
                nota_fp += float((~nota_true & nota_pred).sum().item())
                nota_fn += float((nota_true & ~nota_pred).sum().item())

                known_mask = (y_true < N)
                known_total += float(known_mask.sum().item())
                if known_mask.any():
                    known_right += float((y_pred[known_mask] == y_true[known_mask]).sum().item())

                target_tp += float(((y_true < N) & (y_pred < N) & (y_true == y_pred)).sum().item())
                target_fp += float((((y_true == N) & (y_pred < N)) | ((y_true < N) & (y_pred < N) & (y_true != y_pred))).sum().item())
                target_fn += float((((y_true < N) & (y_pred == N)) | ((y_true < N) & (y_pred < N) & (y_true != y_pred))).sum().item())
                gold_known_count += float((y_true < N).sum().item())
                pred_known_count += float((y_pred < N).sum().item())
                total_count += float(y_true.numel())

                iter_right += self.item(right.data)
                iter_f1_macro += f1_macro
                iter_sample += 1

                nota_prec = nota_tp / max(nota_tp + nota_fp, 1e-12)
                nota_rec = nota_tp / max(nota_tp + nota_fn, 1e-12)
                nota_f1 = 0.0 if (nota_prec + nota_rec) < 1e-12 else (2 * nota_prec * nota_rec / (nota_prec + nota_rec))
                known_acc = known_right / max(known_total, 1e-12)
                target_p, target_r, target_f1 = self._compute_target_micro_metrics(target_tp, target_fp, target_fn)
                gold_known_ratio = gold_known_count / max(total_count, 1e-12)
                pred_known_ratio = pred_known_count / max(total_count, 1e-12)

                _print_if(
                    (it + 1) % max(1, eval_iter // 20) == 0 or (it + 1) == eval_iter,
                    '[EVAL] step: {0:4} | acc: {1:3.2f}%, macro_f1: {2:3.2f}% | '
                    'known_acc: {3:3.2f}% | NOTA P/R/F1: {4:3.2f}/{5:3.2f}/{6:3.2f}% | '
                    'target_micro P/R/F1: {7:3.2f}/{8:3.2f}/{9:3.2f}% | '
                    'gold/pred_known: {10:3.2f}/{11:3.2f}%'.format(
                        it + 1,
                        100 * iter_right / iter_sample,
                        100 * iter_f1_macro / iter_sample,
                        100 * known_acc,
                        100 * nota_prec,
                        100 * nota_rec,
                        100 * nota_f1,
                        100 * target_p,
                        100 * target_r,
                        100 * target_f1,
                        100 * gold_known_ratio,
                        100 * pred_known_ratio,
                    ),
                    end='\r',
                    flush=True,
                )
            if eval_iter > 0:
                print()

        # final summary metrics
        nota_prec = nota_tp / max(nota_tp + nota_fp, 1e-12)
        nota_rec = nota_tp / max(nota_tp + nota_fn, 1e-12)
        nota_f1 = 0.0 if (nota_prec + nota_rec) < 1e-12 else (2 * nota_prec * nota_rec / (nota_prec + nota_rec))
        known_acc = known_right / max(known_total, 1e-12)
        target_p, target_r, target_f1 = self._compute_target_micro_metrics(target_tp, target_fp, target_fn)
        gold_known_ratio = gold_known_count / max(total_count, 1e-12)
        pred_known_ratio = pred_known_count / max(total_count, 1e-12)

        return (
            iter_right / iter_sample,
            iter_f1_macro / iter_sample,
            known_acc,
            nota_prec,
            nota_rec,
            nota_f1,
            target_p,
            target_r,
            target_f1,
            gold_known_ratio,
            pred_known_ratio,
        )

    def test(self,
             model,
             model_name,
             B, N, K, Q,
             eval_iter,
             na_rate=0,
             ckpt=None,
             output_file=None):
        """
        生成 FewRel NOTA 提交文件：
        - 每个 episode 输出 1 个 label（0..N-1），若预测为 NOTA 输出 -1
        - 自动根据 batch 形状推断每个 episode 的 query 数
        """

        all_pred = []

        model.eval()
        if ckpt is None:
            raise ValueError("A checkpoint is required for online evaluation.")
        else:
            if ckpt != 'none':
                state_dict = self.__load_model__(ckpt)['state_dict']
                own_state = model.state_dict()
                for name, param in state_dict.items():
                    if name not in own_state:
                        continue
                    if own_state[name].shape != param.shape:
                        continue
                    own_state[name].copy_(param)
            eval_dataset = self.test_data_loader

        with torch.no_grad():
            for it in tqdm(range(eval_iter)):
                support, query, rel_text = next(eval_dataset)

                if torch.cuda.is_available():
                    for k in support:
                        support[k] = support[k].cuda()
                    for k in query:
                        query[k] = query[k].cuda()
                    for k in rel_text:
                        rel_text[k] = rel_text[k].cuda()

                # ---- 先根据 batch 形状推断 B 和 total_Q_per_ep ----
                key_s = next(iter(support))
                key_q = next(iter(query))
                bs_support = int(support[key_s].size(0))  # = actual_B * N * K
                bs_query = int(query[key_q].size(0))      # = actual_B * total_Q_per_ep

                actual_B = bs_support // (N * K)
                actual_B = max(1, min(actual_B, B))

                total_Q_per_ep = bs_query // actual_B
                total_Q_per_ep = max(1, total_Q_per_ep)

                # 正确的调用：Q 可以仍然传配置里的 Q，total_Q 用真实的 total_Q_per_ep
                out = model(support, query, rel_text, N, K, Q, total_Q_per_ep)
                if isinstance(out, (tuple, list)):
                    pred = out[1]
                else:
                    _, pred = out

                list_pred = pred.view(-1).cpu().tolist()

                # 每个 episode 取第一个 query 的预测作为提交
                for nn in range(actual_B):
                    idx = nn * total_Q_per_ep
                    if idx < len(list_pred):
                        y = list_pred[idx]
                        if y == N:  # 模型里的第 N 类是 NOTA
                            y = -1
                        all_pred.append(y)

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_pred, f)
            print(f"Saved {len(all_pred)} predictions to {output_file}")
