import argparse
import os
import random
import sys

import numpy as np
import torch

from dataset.FewRelDataset import get_loader
from dataset.FewRelOnlineDataset import get_loader_online
from encoder.BertEncoder import BERTSentenceEncoder
from models.bafrc import BaFRC
from models.RoFRC import RoFRC
from toolkit.framework_fewrel import FewShotREFrameworkFewRel


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='./data/fewrel', help='FewRel root')
    parser.add_argument('--train', default='train_wiki', help='train basename')
    parser.add_argument('--val', default='val_wiki', help='val basename')
    parser.add_argument('--test', default='test_wiki', help='test basename')
    parser.add_argument('--pid2name', default='description_pool', help='pid2name basename')
    parser.add_argument('--dataset_name', default='FewRel', help='for display only, kept as FewRel')

    parser.add_argument('--N', default=5, type=int)
    parser.add_argument('--K', default=1, type=int)
    parser.add_argument('--Q', default=1, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--train_iter', default=30000, type=int)
    parser.add_argument('--val_iter', default=1000, type=int)
    parser.add_argument('--test_iter', default=10000, type=int)
    parser.add_argument('--val_step', default=1000, type=int)

    # --model: BaFRC (default) | RoFRC
    parser.add_argument('--model', default='BaFRC',
                        help='model name: BaFRC or RoFRC (MRM is accepted as a legacy alias)')
    parser.add_argument('--encoder', default='bert')
    parser.add_argument('--max_length', default=128, type=int)
    parser.add_argument('--lr', default=2e-5, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--na_rate', default=1, type=int, help='FewRel train NOTA rate')
    parser.add_argument('--grad_iter', default=1, type=int)
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--load_ckpt', default=None)
    parser.add_argument('--save_ckpt', default=None)
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--pretrain_ckpt', default='bert-base-uncased')
    parser.add_argument('--seed', default=5, type=int)
    parser.add_argument('--test_online', action='store_true')
    parser.add_argument('--test_input', default=None)
    parser.add_argument('--test_output', default='pred-5-1.json')
    parser.add_argument('--encoder_ckpt', default=None)

    # BaFRC-specific hyperparams (ignored when model=RoFRC)
    parser.add_argument('--bafrc_margin', dest='bafrc_margin', default=0.15, type=float)
    parser.add_argument('--mrm_margin', dest='bafrc_margin', type=float, help=argparse.SUPPRESS)
    parser.add_argument('--bafrc_gamma', dest='bafrc_gamma', default=3.0, type=float,
                        help='BaFRC boundary loss smoothness γ (Eq.13-14)')
    parser.add_argument('--mrm_gamma', dest='bafrc_gamma', type=float, help=argparse.SUPPRESS)
    parser.add_argument('--radius_quantile', default=0.1, type=float)
    parser.add_argument('--radius_blend_rho', default=0.5, type=float)
    parser.add_argument('--radius_detach', type=lambda x: x.lower() == 'true', default=True,
                        help='detach support/query radius estimation from autograd')
    parser.add_argument('--radius_max', default=10.0, type=float,
                        help='maximum Euclidean radius; <=0 disables the upper clamp')
    parser.add_argument('--use_query_in_radius', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--use_desc_neg', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--bafrc_dist_type', dest='bafrc_dist_type', default='euclidean', type=str)
    parser.add_argument('--mrm_dist_type', dest='bafrc_dist_type', type=str, help=argparse.SUPPRESS)
    parser.add_argument('--use_query_in_boundary', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--radius_update_interval', default=100, type=int)
    parser.add_argument('--bafrc_loss_pos', dest='bafrc_loss_pos',
                        type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--mrm_loss_pos', dest='bafrc_loss_pos', type=lambda x: x.lower() == 'true', help=argparse.SUPPRESS)
    parser.add_argument('--bafrc_loss_neg', dest='bafrc_loss_neg',
                        type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--mrm_loss_neg', dest='bafrc_loss_neg', type=lambda x: x.lower() == 'true', help=argparse.SUPPRESS)

    # std_desc flag:
    #   BaFRC: controls whether prototype uses standardised descriptions
    #   RoFRC: auto-forced to False (RoFRC always uses a single description per relation)
    parser.add_argument('--use_std_desc', type=lambda x: x.lower() == 'true', default=True)

    # RoFRC-specific hyperparam
    parser.add_argument('--lamb_rofrc', default=1e-5, type=float,
                        help='RoFRC lambda (agreement loss weight; currently disabled)')

    opt = parser.parse_args()

    model_name = opt.model.strip()
    if model_name == 'MRM':
        _warn("--model MRM is a legacy alias; using BaFRC.")
        model_name = 'BaFRC'
    if model_name not in ('BaFRC', 'RoFRC'):
        _err(f"Unsupported model '{model_name}'. Choose 'BaFRC' or 'RoFRC'.")
        sys.exit(1)
    if opt.encoder != 'bert':
        raise NotImplementedError

    # RoFRC always uses a single description per relation (use_std_desc must be False).
    # The description content (plain orig / concat orig+std) is controlled by --pid2name.
    if model_name == 'RoFRC' and opt.use_std_desc:
        _warn("RoFRC requires use_std_desc=False (N+1 rel entries). Auto-setting to False.")
        opt.use_std_desc = False

    if not opt.use_std_desc and opt.test_online:
        _warn("use_std_desc=false may mismatch FewRelOnlineDataset relation-text shape.")

    set_seed(opt.seed)
    sentence_encoder = BERTSentenceEncoder(opt.pretrain_ckpt, opt.max_length, opt.encoder_ckpt)

    train_loader = get_loader(
        opt.train, opt.pid2name, sentence_encoder, N=opt.N, K=opt.K, Q=opt.Q, na_rate=opt.na_rate,
        batch_size=opt.batch_size, root=opt.root, use_std_desc=opt.use_std_desc
    )
    val_loader = get_loader(
        opt.val, opt.pid2name, sentence_encoder, N=opt.N, K=opt.K, Q=opt.Q, na_rate=opt.na_rate,
        batch_size=opt.batch_size, root=opt.root, use_std_desc=opt.use_std_desc
    )
    framework = FewShotREFrameworkFewRel(train_loader, val_loader)

    # ---- Model instantiation ----
    if model_name == 'BaFRC':
        model = BaFRC(
            sentence_encoder,
            hidden_size=opt.hidden_size,
            max_len=opt.max_length,
            bafrc_gamma=opt.bafrc_gamma,
            bafrc_margin=opt.bafrc_margin,
            radius_quantile=opt.radius_quantile,
            radius_blend_rho=opt.radius_blend_rho,
            radius_detach=opt.radius_detach,
            radius_max=opt.radius_max,
            use_query_in_radius=opt.use_query_in_radius,
            use_desc_neg=opt.use_desc_neg,
            dist_type=opt.bafrc_dist_type,
            use_query_in_boundary=opt.use_query_in_boundary,
            radius_update_interval=opt.radius_update_interval,
            use_std_desc=opt.use_std_desc,
            bafrc_loss_pos=opt.bafrc_loss_pos,
            bafrc_loss_neg=opt.bafrc_loss_neg,
        )
        prefix = f"BaFRC-FewRel-{opt.N}way-{opt.K}shot-seed{opt.seed}"
    elif model_name == 'RoFRC':
        model = RoFRC(
            sentence_encoder,
            hidden_size=opt.hidden_size,
            max_len=opt.max_length,
            lamb=opt.lamb_rofrc,
        )
        # embed pid2name variant into prefix so checkpoints don't collide
        pid_tag = os.path.splitext(os.path.basename(opt.pid2name))[0]
        prefix = f"RoFRC-FewRel-{opt.N}way-{opt.K}shot-{pid_tag}-seed{opt.seed}"

    if torch.cuda.is_available():
        model.cuda()

    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    ckpt = opt.save_ckpt or f"checkpoint/{prefix}.pth.tar"

    if not opt.only_test:
        opt.train_iter = opt.train_iter * opt.grad_iter
        framework.train(
            model, model_name, opt.batch_size, opt.N, opt.K, opt.Q, opt.na_rate,
            learning_rate=opt.lr, weight_decay=opt.weight_decay,
            train_iter=opt.train_iter, val_iter=opt.val_iter, val_step=opt.val_step,
            load_ckpt=opt.load_ckpt, save_ckpt=ckpt, grad_iter=opt.grad_iter
        )
    elif opt.test_online:
        if not opt.load_ckpt:
            _err("--load_ckpt is required for --test_online")
            return
        if not opt.test_input:
            _err("--test_input is required for --test_online")
            return
        test_loader, num_ep = get_loader_online(
            opt.test_input, opt.pid2name, sentence_encoder,
            N=opt.N, K=opt.K, batch_size=opt.batch_size, root=opt.root
        )
        framework.test_data_loader = test_loader
        eval_iter = (num_ep + opt.batch_size - 1) // opt.batch_size
        framework.test(
            model, model_name, opt.batch_size, opt.N, opt.K, opt.Q,
            eval_iter=eval_iter, na_rate=0, ckpt=opt.load_ckpt, output_file=opt.test_output
        )
        return

    eval_ckpt = opt.load_ckpt if opt.only_test else ckpt
    if not eval_ckpt:
        eval_ckpt = 'none'

    test15_loader = get_loader(
        opt.test, opt.pid2name, sentence_encoder, N=opt.N, K=opt.K, Q=opt.Q, na_rate=1,
        batch_size=opt.batch_size, root=opt.root, use_std_desc=opt.use_std_desc
    )
    test30_loader = get_loader(
        opt.test, opt.pid2name, sentence_encoder, N=opt.N, K=opt.K, Q=opt.Q, na_rate=2,
        batch_size=opt.batch_size, root=opt.root, use_std_desc=opt.use_std_desc
    )
    test50_loader = get_loader(
        opt.test, opt.pid2name, sentence_encoder, N=opt.N, K=opt.K, Q=opt.Q, na_rate=5,
        batch_size=opt.batch_size, root=opt.root, use_std_desc=opt.use_std_desc
    )

    acc_15, f1_15, known_acc_15, nota_p_15, nota_r_15, nota_f1_15, t_p_15, t_r_15, t_f1_15, gk_15, pk_15 = framework.eval(
        model, model_name, opt.batch_size, opt.N, opt.K, opt.Q, na_rate=1,
        eval_iter=opt.test_iter, ckpt=eval_ckpt, test_data_loader=test15_loader
    )
    acc_30, f1_30, known_acc_30, nota_p_30, nota_r_30, nota_f1_30, t_p_30, t_r_30, t_f1_30, gk_30, pk_30 = framework.eval(
        model, model_name, opt.batch_size, opt.N, opt.K, opt.Q, na_rate=2,
        eval_iter=opt.test_iter, ckpt=eval_ckpt, test_data_loader=test30_loader
    )
    acc_50, f1_50, known_acc_50, nota_p_50, nota_r_50, nota_f1_50, t_p_50, t_r_50, t_f1_50, gk_50, pk_50 = framework.eval(
        model, model_name, opt.batch_size, opt.N, opt.K, opt.Q, na_rate=5,
        eval_iter=opt.test_iter, ckpt=eval_ckpt, test_data_loader=test50_loader
    )

    print("[FewRel test na_rate=1] ACC: %.2f, F1: %.2f | known_acc: %.2f | NOTA P/R/F1: %.2f/%.2f/%.2f | "
          "target_micro P/R/F1: %.2f/%.2f/%.2f | gold/pred_known: %.2f/%.2f" % (
              acc_15 * 100, f1_15 * 100, known_acc_15 * 100, nota_p_15 * 100, nota_r_15 * 100, nota_f1_15 * 100,
              t_p_15 * 100, t_r_15 * 100, t_f1_15 * 100, gk_15 * 100, pk_15 * 100
          ))
    print("[FewRel test na_rate=2] ACC: %.2f, F1: %.2f | known_acc: %.2f | NOTA P/R/F1: %.2f/%.2f/%.2f | "
          "target_micro P/R/F1: %.2f/%.2f/%.2f | gold/pred_known: %.2f/%.2f" % (
              acc_30 * 100, f1_30 * 100, known_acc_30 * 100, nota_p_30 * 100, nota_r_30 * 100, nota_f1_30 * 100,
              t_p_30 * 100, t_r_30 * 100, t_f1_30 * 100, gk_30 * 100, pk_30 * 100
          ))
    print("[FewRel test na_rate=5] ACC: %.2f, F1: %.2f | known_acc: %.2f | NOTA P/R/F1: %.2f/%.2f/%.2f | "
          "target_micro P/R/F1: %.2f/%.2f/%.2f | gold/pred_known: %.2f/%.2f" % (
              acc_50 * 100, f1_50 * 100, known_acc_50 * 100, nota_p_50 * 100, nota_r_50 * 100, nota_f1_50 * 100,
              t_p_50 * 100, t_r_50 * 100, t_f1_50 * 100, gk_50 * 100, pk_50 * 100
          ))


if __name__ == '__main__':
    main()
