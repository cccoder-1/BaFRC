import argparse
import csv
import os
import random
import sys

import numpy as np
import torch

from dataset.FSTacredEpisodeDataset import get_loader_fs_tacred
from encoder.BertEncoder import BERTSentenceEncoder
from models.bafrc import BaFRC
from toolkit.framework_tacred import FewShotREFrameworkTacred


def _warn(msg: str) -> None:
    sys.stdout.write("[WARN] " + str(msg) + "\n")
    sys.stdout.flush()


def _err(msg: str) -> None:
    sys.stdout.write("[ERROR] " + str(msg) + "\n")
    sys.stdout.flush()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run_boundary_calibration(
    model,
    test_loader,
    N: int,
    K: int,
    Q: int,
    B: int,
    eval_margins: list,
    shot: int,
    tag: str,
    log_dir: str,
    eval_iter: int,
):
    """
    Inference-time Boundary Calibration.

    For each test episode, runs the BERT encoder exactly once, then applies
    every eval_margin in eval_margins to produce predictions without re-encoding.
    Saves per-margin metrics to {log_dir}/seed{tag}.tsv.
    """
    os.makedirs(log_dir, exist_ok=True)
    tsv_path = os.path.join(log_dir, f"seed{tag}.tsv")

    M = len(eval_margins)
    # Per-margin accumulators
    acc_sum      = [0.0] * M
    nota_tp      = [0.0] * M
    nota_fp      = [0.0] * M
    nota_fn      = [0.0] * M
    known_right  = [0.0] * M
    known_total  = [0.0] * M
    target_tp    = [0.0] * M
    target_fp    = [0.0] * M
    target_fn    = [0.0] * M
    gold_known   = [0.0] * M
    pred_known   = [0.0] * M
    total_count  = [0.0] * M
    n_episodes   = 0

    model.eval()
    with torch.no_grad():
        for it in range(eval_iter):
            support, query, query_label, label, rel_text = next(test_loader)
            if torch.cuda.is_available():
                for k in support:
                    support[k] = support[k].cuda()
                for k in query:
                    query[k] = query[k].cuda()
                for k in rel_text:
                    rel_text[k] = rel_text[k].cuda()
                query_label = query_label.cuda()

            total_Q_ep = query_label.size(0) // B
            query_label = query_label.view(B, total_Q_ep)

            # Single forward pass — encodes support + query once
            model(support, query, rel_text, N, K, Q, total_Q_ep)
            cache = model._cache
            dists_q   = cache["dists_q"]    # (B, total_Q_ep, N)
            support_h = cache["support_h"]  # (B, N, K, d)
            P         = cache["P"]          # (B, N, d)

            # Compute raw quantile radius (before margin subtraction)
            radius_base = model.compute_support_radius_base(support_h, P, N)  # (B, N)

            y_true = query_label.view(-1)

            for mi, eval_margin in enumerate(eval_margins):
                pred   = model.predict_with_eval_margin(dists_q, radius_base, eval_margin, N)
                y_pred = pred.view(-1)

                acc_sum[mi] += float((y_pred == y_true).float().mean())

                nota_true = (y_true == N)
                nota_pred = (y_pred == N)
                nota_tp[mi] += float((nota_true  & nota_pred).sum())
                nota_fp[mi] += float((~nota_true & nota_pred).sum())
                nota_fn[mi] += float((nota_true  & ~nota_pred).sum())

                km = (y_true < N)
                known_total[mi] += float(km.sum())
                if km.any():
                    known_right[mi] += float((y_pred[km] == y_true[km]).sum())

                target_tp[mi] += float(((y_true < N) & (y_pred < N) & (y_true == y_pred)).sum())
                target_fp[mi] += float((
                    ((y_true == N) & (y_pred < N)) |
                    ((y_true < N) & (y_pred < N) & (y_true != y_pred))
                ).sum())
                target_fn[mi] += float((
                    ((y_true < N) & (y_pred == N)) |
                    ((y_true < N) & (y_pred < N) & (y_true != y_pred))
                ).sum())
                gold_known[mi]  += float((y_true < N).sum())
                pred_known[mi]  += float((y_pred < N).sum())
                total_count[mi] += float(y_true.numel())

            n_episodes += 1
            if (it + 1) % 1000 == 0:
                print(f"\r[Calibration] {it+1}/{eval_iter} episodes ...", end='', flush=True)

    print(f"\n[Calibration] Computing final metrics over {n_episodes} episodes.")

    results = []
    for mi, eval_margin in enumerate(eval_margins):
        ns  = max(n_episodes, 1)
        acc = acc_sum[mi] / ns

        nota_p  = nota_tp[mi] / max(nota_tp[mi] + nota_fp[mi], 1e-12)
        nota_r  = nota_tp[mi] / max(nota_tp[mi] + nota_fn[mi], 1e-12)
        nota_f1 = 0.0 if (nota_p + nota_r) < 1e-12 else 2 * nota_p * nota_r / (nota_p + nota_r)

        ka   = known_right[mi] / max(known_total[mi], 1e-12)
        tp, fp, fn = target_tp[mi], target_fp[mi], target_fn[mi]
        t_p  = tp / max(tp + fp, 1e-12)
        t_r  = tp / max(tp + fn, 1e-12)
        t_f1 = 0.0 if (t_p + t_r) < 1e-12 else 2 * t_p * t_r / (t_p + t_r)

        gk      = gold_known[mi] / max(total_count[mi], 1e-12)
        pk      = pred_known[mi] / max(total_count[mi], 1e-12)
        nota_far = 1.0 - nota_r

        print(
            f"\n[Calibration]\n"
            f"  margin={eval_margin:.2f}\n"
            f"  ACC={acc*100:.2f}  known_acc={ka*100:.2f}\n"
            f"  target_P/R/F1={t_p*100:.2f}/{t_r*100:.2f}/{t_f1*100:.2f}\n"
            f"  NOTA_P/R/F1={nota_p*100:.2f}/{nota_r*100:.2f}/{nota_f1*100:.2f}\n"
            f"  NOTA_FAR={nota_far*100:.2f}\n"
            f"  gold/pred_known={gk*100:.2f}/{pk*100:.2f}"
        )

        results.append({
            "shot": shot, "seed": tag, "margin": f"{eval_margin:.4f}",
            "acc": f"{acc*100:.4f}",
            "known_acc": f"{ka*100:.4f}",
            "target_p": f"{t_p*100:.4f}",
            "target_r": f"{t_r*100:.4f}",
            "target_f1": f"{t_f1*100:.4f}",
            "nota_p": f"{nota_p*100:.4f}",
            "nota_r": f"{nota_r*100:.4f}",
            "nota_f1": f"{nota_f1*100:.4f}",
            "nota_far": f"{nota_far*100:.4f}",
            "gold_known_ratio": f"{gk*100:.4f}",
            "pred_known_ratio": f"{pk*100:.4f}",
        })

    fieldnames = [
        "shot", "seed", "margin", "acc", "known_acc",
        "target_p", "target_r", "target_f1",
        "nota_p", "nota_r", "nota_f1", "nota_far",
        "gold_known_ratio", "pred_known_ratio",
    ]
    with open(tsv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[Calibration] Results saved to {tsv_path}")

    # Sanity check: verify that margin sweep actually changes predictions
    pk_vals = [float(r["pred_known_ratio"]) for r in results]
    if len(set(round(v, 3) for v in pk_vals)) == 1:
        print(
            "[WARNING] All eval_margin values produced identical pred_known_ratio! "
            "Check that eval_margin is being applied in predict_with_eval_margin()."
        )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='./data/fs_tacred', help='FS-TACRED root')
    parser.add_argument('--train', default='fs_tacred_train_5way_1shot_seed5', help='train file basename')
    parser.add_argument('--val', default='fs_tacred_dev_5way_1shot_seed5', help='val file basename')
    parser.add_argument('--test', default='fs_tacred_test_5way_1shot_seed5', help='test file basename')
    parser.add_argument('--pid2name', default='description_pool', help='pid2name basename')

    parser.add_argument('--N', default=5, type=int, help='N way')
    parser.add_argument('--K', default=1, type=int, help='K shot')
    parser.add_argument('--Q', default=1, type=int, help='query instances hint')
    parser.add_argument('--batch_size', default=2, type=int, help='batch size')
    parser.add_argument(
        '--train_known_ratio',
        default=0.0,
        type=float,
        help='target known-episode ratio for training-only weighted sampling; 0 keeps original order',
    )
    parser.add_argument('--train_iter', default=30000, type=int, help='num of iters in training')
    parser.add_argument('--val_iter', default=1000, type=int, help='num of iters in validation')
    parser.add_argument('--test_iter', default=10000, type=int, help='num of iters in testing')
    parser.add_argument('--val_step', default=1000, type=int, help='val every val_step')
    parser.add_argument(
        '--early_stopping_patience',
        default=6,
        type=int,
        help='number of consecutive non-improving validations; <=0 disables early stopping',
    )

    parser.add_argument('--model', default='BaFRC', help='only BaFRC is supported (MRM is accepted as a legacy alias)')
    parser.add_argument('--encoder', default='bert', help='encoder: bert')
    parser.add_argument('--max_length', default=128, type=int, help='max length')
    parser.add_argument('--lr', default=2e-5, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay')
    parser.add_argument('--na_rate', default=0, type=int, help='ignored for FS-TACRED, always 0')
    parser.add_argument('--grad_iter', default=1, type=int, help='accumulate gradient every x iterations')
    parser.add_argument('--hidden_size', default=768, type=int, help='hidden size')
    parser.add_argument('--load_ckpt', default=None, help='load ckpt')
    parser.add_argument('--save_ckpt', default=None, help='save ckpt')
    parser.add_argument('--only_test', action='store_true', help='only test')
    parser.add_argument('--boundary_calibration', action='store_true',
                        help='Inference-time Boundary Calibration: sweep eval_margin_list on a fixed checkpoint')
    parser.add_argument('--eval_margin_list', default='0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.40',
                        type=str, help='comma-separated eval margins for boundary calibration')
    parser.add_argument('--calibration_tag', default=None,
                        help='tag for calibration TSV filename (default: opt.seed); '
                             'set to test-episode seed in shell scripts to avoid overwrite')
    parser.add_argument('--pretrain_ckpt', default='bert-base-uncased', help='bert pre-trained checkpoint')
    parser.add_argument('--seed', default=5, type=int, help='seed')
    parser.add_argument('--encoder_ckpt', default=None, help='encoder-only ckpt for BertEncoder')

    parser.add_argument('--bafrc_margin', dest='bafrc_margin', default=0.15, type=float)
    parser.add_argument('--mrm_margin', dest='bafrc_margin', type=float, help=argparse.SUPPRESS)
    parser.add_argument('--bafrc_gamma', dest='bafrc_gamma', default=3.0, type=float)
    parser.add_argument('--mrm_gamma', dest='bafrc_gamma', type=float, help=argparse.SUPPRESS)
    parser.add_argument('--radius_quantile', default=0.1, type=float)
    parser.add_argument(
        '--radius_blend_rho',
        default=0.5,
        type=float,
        help='blend rho for R_train = (1-rho)*R_sup + rho*R_qry (see BaFRC.loss)',
    )
    parser.add_argument(
        '--radius_max',
        default=0.0,
        type=float,
        help='maximum Euclidean radius; <=0 disables the upper clamp',
    )
    parser.add_argument('--use_query_in_radius', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--use_desc_neg', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--bafrc_dist_type', dest='bafrc_dist_type', default='euclidean', type=str)
    parser.add_argument('--mrm_dist_type', dest='bafrc_dist_type', type=str, help=argparse.SUPPRESS)
    parser.add_argument('--use_query_in_boundary', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--radius_update_interval', default=100, type=int)
    parser.add_argument('--use_std_desc', type=lambda x: x.lower() == 'true', default=True)
    opt = parser.parse_args()

    model_name = opt.model.strip()
    if model_name == 'MRM':
        _warn("--model MRM is a legacy alias; using BaFRC.")
        model_name = 'BaFRC'
    if model_name != 'BaFRC':
        _err("Only BaFRC is supported in train_tacred.py")
        sys.exit(1)
    if opt.encoder != 'bert':
        raise NotImplementedError
    if opt.na_rate != 0:
        _warn("FS-TACRED uses pre-sampled episodes; force --na_rate=0.")
        opt.na_rate = 0
    if not 0.0 <= opt.train_known_ratio <= 1.0:
        _err("--train_known_ratio must be in [0, 1].")
        sys.exit(1)

    if opt.boundary_calibration:
        if not opt.only_test:
            _err("--boundary_calibration requires --only_test (no retraining).")
            sys.exit(1)
        if not opt.load_ckpt:
            _err("--boundary_calibration requires --load_ckpt.")
            sys.exit(1)

    set_seed(opt.seed)
    sentence_encoder = BERTSentenceEncoder(opt.pretrain_ckpt, opt.max_length, opt.encoder_ckpt)

    test_loader = get_loader_fs_tacred(
        opt.test, opt.pid2name, sentence_encoder, N=opt.N, K=opt.K, Q=opt.Q, na_rate=0,
        batch_size=opt.batch_size, root=opt.root, use_std_desc=opt.use_std_desc
    )

    model = BaFRC(
        sentence_encoder,
        hidden_size=opt.hidden_size,
        max_len=opt.max_length,
        bafrc_gamma=opt.bafrc_gamma,
        bafrc_margin=opt.bafrc_margin,
        radius_quantile=opt.radius_quantile,
        radius_blend_rho=opt.radius_blend_rho,
        radius_max=opt.radius_max,
        use_query_in_radius=opt.use_query_in_radius,
        use_desc_neg=opt.use_desc_neg,
        dist_type=opt.bafrc_dist_type,
        use_query_in_boundary=opt.use_query_in_boundary,
        radius_update_interval=opt.radius_update_interval,
        use_std_desc=opt.use_std_desc,
    )
    if torch.cuda.is_available():
        model.cuda()

    # ------------------------------------------------------------------
    # Inference-time Boundary Calibration (no training, no self.margin change)
    # ------------------------------------------------------------------
    if opt.boundary_calibration:
        eval_margins = [float(x.strip()) for x in opt.eval_margin_list.split(',') if x.strip()]
        if not eval_margins:
            _err("--eval_margin_list is empty.")
            sys.exit(1)

        # Load checkpoint once
        state_dict = torch.load(opt.load_ckpt, map_location='cpu')['state_dict']
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                own_state[name].copy_(param)
        print(f"Successfully loaded checkpoint '{opt.load_ckpt}'")

        tag = opt.calibration_tag if opt.calibration_tag is not None else str(opt.seed)
        log_dir = f"log/calibration/tacred/{opt.N}way_{opt.K}shot"
        print(
            f"[Calibration] Inference-time Boundary Calibration | "
            f"margins={eval_margins} | tag={tag} | log_dir={log_dir}"
        )
        run_boundary_calibration(
            model=model,
            test_loader=test_loader,
            N=opt.N, K=opt.K, Q=opt.Q, B=opt.batch_size,
            eval_margins=eval_margins,
            shot=opt.K,
            tag=tag,
            log_dir=log_dir,
            eval_iter=opt.test_iter,
        )
        return

    # ------------------------------------------------------------------
    # Normal train / test path (unchanged)
    # ------------------------------------------------------------------
    train_loader = get_loader_fs_tacred(
        opt.train, opt.pid2name, sentence_encoder, N=opt.N, K=opt.K, Q=opt.Q, na_rate=0,
        batch_size=opt.batch_size, root=opt.root, use_std_desc=opt.use_std_desc,
        train_known_ratio=opt.train_known_ratio, sampling_seed=opt.seed
    )
    val_loader = get_loader_fs_tacred(
        opt.val, opt.pid2name, sentence_encoder, N=opt.N, K=opt.K, Q=opt.Q, na_rate=0,
        batch_size=opt.batch_size, root=opt.root, use_std_desc=opt.use_std_desc
    )
    framework = FewShotREFrameworkTacred(train_loader, val_loader)

    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    prefix = f"BaFRC-FS_TACRED-{opt.N}way-{opt.K}shot-seed{opt.seed}"
    ckpt = opt.save_ckpt or f"checkpoint/{prefix}.pth.tar"

    if not opt.only_test:
        opt.train_iter = opt.train_iter * opt.grad_iter
        framework.train(
            model, model_name, opt.batch_size, opt.N, opt.K, opt.Q, 0,
            learning_rate=opt.lr, weight_decay=opt.weight_decay,
            train_iter=opt.train_iter, val_iter=opt.val_iter, val_step=opt.val_step,
            load_ckpt=opt.load_ckpt, save_ckpt=ckpt, grad_iter=opt.grad_iter,
            early_stopping_patience=opt.early_stopping_patience,
        )
        eval_ckpt = ckpt
    else:
        eval_ckpt = opt.load_ckpt or 'none'
        if eval_ckpt == 'none':
            _warn("No --load_ckpt provided, evaluating pre-trained backbone init.")

    acc, f1, known_acc, nota_p, nota_r, nota_f1, t_p, t_r, t_f1, gk_ratio, pk_ratio = framework.eval(
        model, model_name, opt.batch_size, opt.N, opt.K, opt.Q, na_rate=0,
        eval_iter=opt.test_iter, ckpt=eval_ckpt, test_data_loader=test_loader
    )
    print(
        "[FS_TACRED test] ACC: %.2f, F1: %.2f | known_acc: %.2f | NOTA P/R/F1: %.2f/%.2f/%.2f | "
        "target_micro P/R/F1: %.2f/%.2f/%.2f | gold/pred_known: %.2f/%.2f"
        % (
            acc * 100, f1 * 100, known_acc * 100, nota_p * 100, nota_r * 100, nota_f1 * 100,
            t_p * 100, t_r * 100, t_f1 * 100, gk_ratio * 100, pk_ratio * 100
        )
    )


if __name__ == '__main__':
    main()
