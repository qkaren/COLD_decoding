#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import time
import wandb
import sys
sys.path.insert(0, './GPT2ForwardBackward')

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import tokenize

from util import *
from util import _constraint_loss2
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from bleuloss import batch_log_bleulosscnn_ae
from modeling_opengpt2 import OpenGPT2LMHeadModel
from padded_encoder import Encoder


stop_words = set(stopwords.words('english'))


def options():
    parser = argparse.ArgumentParser()
    ## setting
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--no-cuda", action="store_true", help="no cuda")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--print-every", type=int, default=200)
    parser.add_argument("--pretrained_model", type=str, default="gpt2-large")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--straight-through", action="store_true")
    parser.add_argument("--topk", type=int, default=0)
    parser.add_argument("--rl-topk", type=int, default=0)
    parser.add_argument("--lexical", type=str, default='max', choices=['max', 'ppl_max', 'all', 'bleu'])
    parser.add_argument("--lexical-variants", action="store_true", help="")
    ## experiment
    parser.add_argument("--input-file", type=str,
                        default="./data/commongen/commongen_data/test.multi.constraint.json")
    parser.add_argument("--output-dir", type=str, default="./data/commongen/")
    parser.add_argument("--fwd-model", type=str,
                        default="/var/karen/workspace/GPT2ForwardBackward/opengpt2_pytorch_forward")
    parser.add_argument("--back-model", type=str,
                        default="danyaljj/opengpt2_pytorch_backward")
    parser.add_argument("--version", type=str, default="")
    parser.add_argument("--start", type=int, default=1, help="loading data from ith examples.")
    parser.add_argument("--end", type=int, default=10, help="loading data util ith examples.")
    parser.add_argument("--mode", type=str, default='constrained_langevin',
                        choices=['constrained_langevin', 'counterfactual_langevin', 'abductive_langevin', 'grammar'])
    ## model
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--length", type=int, default=15, help="maximum length of optimized logits.")
    parser.add_argument("--max-length", type=int, default=50, help="maximum length of complete sentence.")
    parser.add_argument("--frozen-length", type=int, default=0, help="length of optimization window in sequence.")
    parser.add_argument("--constraint-weight", type=float, default=0.1)
    parser.add_argument("--abductive-c2-weight", type=float, default=0.05)
    parser.add_argument("--abductive-filterx", action="store_true", help="filter out keywords included in x")
    parser.add_argument("--lr-nll-portion", type=float, default=1)
    parser.add_argument("--prefix-length", type=int, default=0, help="length of prefix.")
    parser.add_argument("--counterfactual-max-ngram", type=int, default=6)
    parser.add_argument("--no-loss-rerank", action="store_true", help="")
    # temperature
    parser.add_argument("--input-lgt-temp", type=float, default=1,
                        help="temperature of logits used for model input.")
    parser.add_argument("--output-lgt-temp", type=float, default=1,
                        help="temperature of logits used for model output.")
    parser.add_argument("--rl-output-lgt-temp", type=float, default=1,
                        help="temperature of logits used for model output.")
    parser.add_argument("--init-temp", type=float, default=0.1,
                        help="temperature of logits used in the initialization pass. High => uniform init.")
    parser.add_argument("--init-mode", type=str, default='random', choices=['random', 'original'])
    # lr
    parser.add_argument("--stepsize", type=float, default=0.1, help="learning rate in the backward pass.")
    parser.add_argument("--stepsize-ratio", type=float, default=1, help="")
    parser.add_argument("--stepsize-iters", type=int, default=1000, help="")
    # iterations
    parser.add_argument("--num-iters", type=int, default=1000)
    parser.add_argument("--min-iters", type=int, default=0, help="record best only after N iterations")
    parser.add_argument("--noise-iters", type=int, default=1, help="add noise at every N iterations")
    parser.add_argument("--win-anneal-iters", type=int, default=-1, help="froze the optimization window after N iters")
    parser.add_argument("--constraint-iters", type=int, default=1000,
                        help="add one more group of constraints from N iters")
    # gaussian noise
    parser.add_argument("--gs_mean", type=float, default=0.0)
    parser.add_argument("--gs_std", type=float, default=0.01)
    parser.add_argument("--large-noise-iters", type=str, default="-1", help="Example: '50,1000'")
    parser.add_argument("--large_gs_std", type=str, default="1", help="Example: '1,0.1'")

    args = parser.parse_args()
    return args


def decode(model, tokenizer, device, x="", z="", constraints=None, args=None, model_back=None, zz=None):
    '''
    x: left context   (prompt in lexical constrained task)
    z: optimization target  (original ending in counterfactual task)
    constraints: (constraint set in lexical constrained task)
    '''

    x_ = tokenizer.encode(x)
    x_t = torch.tensor(x_, device=device, dtype=torch.long)
    x_onehot = one_hot(x_t, dimension=tokenizer.vocab_size)

    # repeat batch_size times
    x_t = x_t.unsqueeze(0).repeat(args.batch_size, 1)
    x_onehot = x_onehot.repeat(args.batch_size, 1, 1)

    z_mask = None

    if 'counterfactual' in args.mode:
        z_ = tokenizer.encode(z)[1:]  # delete the "." token we appended before
        z_t = torch.tensor(z_, device=device, dtype=torch.long)

        z_onehot = one_hot(z_t, dimension=tokenizer.vocab_size)
        z_onehot = z_onehot.repeat(args.batch_size, 1, 1)

        z_t = z_t.unsqueeze(0).repeat(args.batch_size, 1)

        length = args.length
        if length <= 0:
            length = z_t.shape[1] - length
        if args.verbose:
            print("x:\t|%s|\nz:\t|%s|\nlength:\t%d\nconstraints:\t%s" % (
                tokenizer.decode(x_), tokenizer.decode(z_), length, constraints))

        z_words = word_tokenize(z[2:])  # delete the ". " token we appended before
        z_nonstop_words = [w.lower() for w in z_words if w.lower() not in stop_words and w.isalnum()]
        z_nonstop_words += [z_words[0]]  # add the first token
        z_nonstop_words = ' ' + ' '.join(z_nonstop_words)
        z_nonstop_ = tokenizer.encode(z_nonstop_words)
        print('|' + z_nonstop_words + '|')

        z_mask = np.zeros([tokenizer.vocab_size])
        z_mask[z_nonstop_] = 1.
        z_mask = torch.tensor(z_mask, device=device)
        z_mask = z_mask.unsqueeze(0).unsqueeze(0).repeat(args.batch_size, length, 1)

    if 'abductive' in args.mode:
        length = args.length

        z_ = tokenizer.encode(z)[1:]  # delete the "." token we appended before
        z_t = torch.tensor(z_, device=device, dtype=torch.long)
        z_onehot = one_hot(z_t, dimension=tokenizer.vocab_size)
        # repeat batch_size times
        z_t = z_t.unsqueeze(0).repeat(args.batch_size, 1)
        z_onehot = z_onehot.repeat(args.batch_size, 1, 1)

        zz_ = tokenizer.encode(zz)[1:]  # delete the "." token we appended before
        zz_t = torch.tensor(zz_, device=device, dtype=torch.long)
        zz_t = zz_t.unsqueeze(0).repeat(args.batch_size, 1)

        z_mask = np.zeros([tokenizer.vocab_size])
        z_mask[zz_] = 1.
        z_mask = torch.tensor(z_mask, device=device)
        z_mask = z_mask.unsqueeze(0).unsqueeze(0).repeat(args.batch_size, length, 1)

        if args.verbose:
            print("x:\t|%s|\nz:\t|%s|\nzz:\t|%s|\nconstraints:\t%s" % (
                tokenizer.decode(x_), tokenizer.decode(z_), tokenizer.decode(zz_), constraints))

    cs_ = None
    cs_onehot = None
    if 'lexical' in args.mode:
        z_t = None
        length = args.length
        if args.lexical == 'bleu':
            assert not args.lexical_variants
            z = ' '.join([c.strip() for c in constraints])
            z = ' ' + z.strip()
            z_ = tokenizer.encode(z)
            z_t = torch.tensor(z_, device=device, dtype=torch.long)
            z_t = z_t.unsqueeze(0).repeat(args.batch_size, 1)
            if args.verbose:
                print("x:\t|%s|\nz:\t|%s|" % (tokenizer.decode(x_), tokenizer.decode(z_)))

        if args.lexical_variants:
            cs_ = []
            cs_onehot = []
            cs_list = []
            for c_v in constraints:  # variants of a constraint token
                cs = []
                for c in c_v:
                    c_ = tokenizer.encode(c)
                    for c_tok in c_:
                        cs.append(c_tok)
                cs_list += cs
                cur_cs_ = torch.tensor(cs, device=device, dtype=torch.long)
                cur_cs_ = cur_cs_.unsqueeze(0)
                cur_cs_onehot = one_hot(cur_cs_, dimension=tokenizer.vocab_size)
                cs_.append(cur_cs_)
                cs_onehot.append(cur_cs_onehot)
            if args.verbose:
                print("x:\t%s\nconstraints:\t%s" % (tokenizer.decode(x_), constraints))
                print(cs_)
        else:
            cs = []
            for c in constraints:
                if args.lexical == 'bleu':
                    c_ = tokenizer.encode(" " + c)
                else:
                    c_ = tokenizer.encode(c)
                for c_tok in c_:
                    cs.append(c_tok)
            cs_list = cs
            cs_ = torch.tensor(cs, device=device, dtype=torch.long)
            cs_ = cs_.unsqueeze(0)
            cs_onehot = one_hot(cs_, dimension=tokenizer.vocab_size)
            if args.verbose:
                print("x:\t%s\nconstraints:\t%s" % (tokenizer.decode(x_), constraints))

        z_mask = np.zeros([tokenizer.vocab_size])
        z_mask[cs_list] = 1.
        z_mask = torch.tensor(z_mask, device=device)
        z_mask = z_mask.unsqueeze(0).unsqueeze(0).repeat(args.batch_size, length, 1)

    model.eval()

    if args.init_mode == 'random':
        init_logits = initialize(model, x_t, length, args.init_temp, device)
    else:
        init_logits = z_onehot / 0.1
        init_logits = init_logits[:, :length, :]
        if length > init_logits.shape[1]:
            init_logits = torch.cat(
                [init_logits,
                 torch.zeros([args.batch_size, length - init_logits.shape[1], tokenizer.vocab_size], device=device)],
                dim=1)
    text, _, _ = get_text_from_logits(init_logits, tokenizer)
    for bi in range(args.batch_size):
        print("[initial]: %s" % (text[bi]))

    if args.wandb:
        wandb.init(
            project='args.mode' + str(int(round(time.time() * 1000))),
            config=args)

    assert args.prefix_length <= 0  # Otherwise not compatible with batch mode

    if args.prefix_length > 0:
        prefix_logits = torch.nn.Parameter(
            torch.rand(x_onehot.shape[0], args.prefix_length, x_onehot.shape[2], dtype=init_logits.dtype,
                       device=device))

    y_logits = init_logits
    epsilon = torch.nn.Parameter(torch.zeros_like(y_logits))
    if args.prefix_length > 0:
        optim = torch.optim.Adam([epsilon, prefix_logits], lr=args.stepsize)
    else:
        optim = torch.optim.Adam([epsilon], lr=args.stepsize)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, step_size=args.stepsize_iters,
                                                gamma=args.stepsize_ratio)

    frozen_len = args.frozen_length

    y_logits_ = None
    noise_std = 0.0

    ## Encode x beforehand
    assert args.prefix_length <= 0, "The current code does not support prefix-length > 0"
    soft_forward_x = x_onehot[:, -1:, :]  # The last token of x is used in soft_forward
    if x_t.shape[1] == 1:
        x_model_past = None
    else:
        x_model_outputs = model(x_t[:, :-1])
        x_model_past = x_model_outputs.past_key_values
        x_model_past = [_.detach() for _ in x_model_past]

    # For right to left model
    rl_reverse_index = torch.arange(y_logits.shape[1] - 1, -1, -1)

    mask_t = None

    for iter in range(args.num_iters):
        optim.zero_grad()
        y_logits_ = y_logits + epsilon

        soft_forward_y = y_logits_ / 0.001
        if args.straight_through:
            if mask_t is None:
                soft_forward_y = (y_logits_.detach() / 0.001 - y_logits_).detach() + y_logits_
            else:
                soft_forward_y = top_k_filter_3d(y_logits_, args.topk, mask=mask_t, extra_mask=z_mask) / 0.001

        y_logits_t = soft_forward(model, soft_forward_x, soft_forward_y, x_past=x_model_past)

        if args.topk == 0:
            mask_t = None
        else:
            _, indices_t = torch.topk(y_logits_t, args.topk)
            mask_t = torch.zeros_like(y_logits_t).scatter_(2, indices_t, 1)

        # Compute loss, gradients, and update.
        lr_nll_loss = soft_nll(
            top_k_filter_3d(y_logits_t / args.output_lgt_temp, args.topk, extra_mask=z_mask),
            y_logits_ / args.input_lgt_temp)

        if args.lr_nll_portion == 1.0:
            rl_nll_loss = lr_nll_loss
        else:
            # add right-to-left model (rl)
            if "lexical" in args.mode or "counterfactual" in args.mode:
                y_logits_rev = y_logits_[:, rl_reverse_index, :]
                y_logits_rev_t = model_back(y_logits_rev.argmax(-1) + 1).logits[:, :-1, :]
                y_logits_rev_t = y_logits_rev_t[:, :, 1:y_logits_.shape[-1] + 1]
                rl_nll_loss = soft_nll(
                    top_k_filter_3d(y_logits_rev_t / args.output_lgt_temp, args.rl_topk),
                    y_logits_rev[:, 1:] / args.input_lgt_temp)
            elif "abductive" in args.mode:
                yz_logits_rev = torch.flip(torch.cat([y_logits_, z_onehot], dim=1), [1])
                yz_logits_rev_t = soft_backward(model_back, yz_logits_rev / 0.00001)
                yz_logits_rev_rev_t = torch.flip(yz_logits_rev_t, [1])
                yz_logits_rev_rev_t = yz_logits_rev_rev_t[:, :, 1:y_logits_.shape[-1] + 1]
                yz_logits_rev_rev_t_ = yz_logits_rev_rev_t[:, :y_logits_.shape[1], :]

                tmp_logits = yz_logits_rev_rev_t_
                repetition_mask = torch.cat([F.softmax(tmp_logits[:, 1:, :], dim=-1),
                                             torch.zeros_like(tmp_logits[:, -1:, :])], dim=1)
                yz_logits_rev_rev_t_ = yz_logits_rev_rev_t_ - repetition_mask * 1e4
                yz_logits_rev_rev_t_ = yz_logits_rev_rev_t_.detach()

                rl_nll_loss = soft_nll(
                    top_k_filter_3d(yz_logits_rev_rev_t_ / args.rl_output_lgt_temp, args.rl_topk),
                    y_logits_ / args.input_lgt_temp)

        if "lexical" in args.mode:
            if args.constraint_iters == 1:
                num_cs = len(cs_)
            else:
                num_cs = iter // args.constraint_iters
            if num_cs <= 0:
                c_loss = lr_nll_loss
            else:
                if args.lexical_variants:
                    cs_onehot_ = cs_onehot[:num_cs]
                    cs_t_ = cs_[:num_cs]
                    if args.lexical == 'max':
                        raise NotImplementedError
                        c_loss = constraint_loss_with_variants(y_logits_, cs_onehot_, cs_t_)
                    elif args.lexical == 'ppl_max':
                        probs_t = y_logits_t.softmax(-1)
                        c_loss = constraint_loss_with_variants_by_ppl(y_logits_, cs_onehot_, cs_t_, probs_t)
                    else:
                        raise NotImplementedError
                elif args.lexical == 'bleu':
                    c_loss = batch_log_bleulosscnn_ae(
                        decoder_outputs=top_k_filter_3d(y_logits_, args.topk, mask=mask_t, extra_mask=z_mask).transpose(
                            0, 1),
                        target_idx=z_t,
                        ngram_list=[1]
                    )
                else:
                    cs_onehot_ = cs_onehot[:, :num_cs]
                    cs_t_ = cs_[:, :num_cs]
                    if args.lexical == 'max':
                        c_loss = constraint_loss(y_logits_, cs_onehot_, cs_t_)
                    elif args.lexical == 'ppl_max':
                        c_loss = constraint_loss_by_ppl(y_logits_, cs_onehot_, cs_t_, y_logits_t)
                    elif args.lexical == 'all':
                        c_loss = constraint_loss_all(y_logits_, cs_onehot_, cs_t_)
                    elif args.lexical == 'bleu':
                        c_loss = batch_log_bleulosscnn_ae(
                            decoder_outputs=y_logits_.transpose(0, 1),
                            target_idx=z_t,
                            ngram_list=[1]
                        )
                    else:
                        raise NotImplementedError

        if "counterfactual" in args.mode:
            c_loss = batch_log_bleulosscnn_ae(
                decoder_outputs=top_k_filter_3d(y_logits_, args.topk, mask=mask_t, extra_mask=z_mask).transpose(0, 1),
                target_idx=z_t,
                ngram_list=list(range(2, args.counterfactual_max_ngram + 1))
            )

        if "abductive" in args.mode:
            soft_forward_y_ = (y_logits_.detach() / 0.3 - y_logits_).detach() + y_logits_
            xyz_logits, xy_length = soft_forward_xyz(model, soft_forward_x, soft_forward_y_, z_onehot)

            # Reshaping
            bz = args.batch_size
            lg = xyz_logits.shape[1]
            st = xy_length - 1
            ed = xyz_logits.shape[1] - 1
            xyz_logits = xyz_logits.view(-1, xyz_logits.shape[-1])
            z_logits = torch.cat([xyz_logits[bi * lg + st:bi * lg + ed, :] for bi in range(bz)], dim=0)

            c_loss_1 = torch.nn.CrossEntropyLoss(reduction='none')(
                z_logits,
                z_t.view(-1))
            c_loss_1 = c_loss_1.view(args.batch_size, -1).mean(-1)

            c_loss_2 = batch_log_bleulosscnn_ae(
                decoder_outputs=y_logits_.transpose(0, 1),
                target_idx=zz_t,
                ngram_list=[1]
            )
            c_loss = c_loss_1 + args.abductive_c2_weight * c_loss_2

        loss = (1.0 - args.constraint_weight) * args.lr_nll_portion * lr_nll_loss \
               + (1.0 - args.constraint_weight) * (1 - args.lr_nll_portion) * rl_nll_loss \
               + args.constraint_weight * c_loss
        loss = loss.mean()

        if iter < args.num_iters - 1:  # so that the mask_t at the last iteration will not change
            loss.backward()
            optim.step()
            scheduler.step()  # turn off the scheduler
            last_lr = scheduler.get_last_lr()[0]

        if args.verbose and ((iter + 1) % args.print_every == 0 or iter == 0 or iter + 1 == args.num_iters):
            text, _, _ = decode_with_model_topk(
                model, y_logits_, args.topk, soft_forward_x, x_model_past, tokenizer, extra_mask=z_mask)
            for bi in range(args.batch_size):
                if "abductive" in args.mode:
                    print(
                        "%d, loss: %.4f, lr_nll_loss: %.4f, rl_nll_loss: %.4f,  c_loss_2: %.4f, lr: %.4f, |%s|" % (
                            iter + 1, loss.item(), lr_nll_loss[bi].item(), rl_nll_loss[bi].item(),
                            c_loss_2[bi].item(), last_lr, text[bi]))
                else:
                    print("%d, loss: %.4f, lr_nll_loss: %.4f, c_loss: %.4f, lr: %.4f, |%s|" % (
                    iter + 1, loss.item(), lr_nll_loss[bi].item(), c_loss[bi].item(), last_lr, text[bi]))

            if "abductive" in args.mode:
                pass

            print()

        if args.wandb:
            wandb.log(
                {"Loss": loss.item(),
                 "left-to-right nll loss": lr_nll_loss.item(),
                 "right-to-left nll loss": rl_nll_loss.item(),
                 "constraint loss": c_loss,
                 "Gassian_Noise_STD": noise_std,
                 "LR": last_lr,
                 "Gradient": torch.norm(epsilon.grad).detach().clone().data.cpu().numpy()}
            )

        ## noise
        if iter < args.num_iters - 1:

            if 'grammar' in args.mode:
                continue

            large_noise_iters = [int(_) for _ in args.large_noise_iters.split(',')]
            large_gs_stds = [float(_) for _ in args.large_gs_std.split(',')]
            noise_std = 0.
            if iter % args.noise_iters == 0:
                noise_last = True
                for ni in range(len(large_noise_iters)):
                    if iter < large_noise_iters[ni]:
                        noise_last = False
                        break
                if noise_last:
                    noise_std = args.gs_std
                else:
                    noise_std = large_gs_stds[ni]

                noise = torch.normal(mean=args.gs_mean, std=noise_std, size=epsilon.size(),
                                     device='cuda', requires_grad=False)
                if args.win_anneal_iters >= 0 and iter >= args.win_anneal_iters:
                    zeros = torch.zeros_like(noise)
                    noise_mix = torch.cat([zeros[:, :frozen_len], noise[:, frozen_len:]], dim=1)
                    y_logits = y_logits + noise_mix
                else:
                    y_logits = y_logits + noise

    if args.wandb:
        wandb.finish()

    text, _, last_text_ids = decode_with_model_topk(
        model, y_logits_, args.topk, soft_forward_x, x_model_past, tokenizer, extra_mask=z_mask)
    if "lexical" in args.mode:
        last_text_ids = torch.cat([x_t, last_text_ids], dim=1)
    last_rank_loss = model(input_ids=last_text_ids, labels=last_text_ids).loss
    last_rank_loss = last_rank_loss.detach().clone().data.cpu().numpy()
    text_post = post_process(last_text_ids, model, args.max_length, args.length, tokenizer, device)
    ppl_last = np.exp(last_rank_loss)

    if args.verbose:
        for bi in range(args.batch_size):
            print("[final]: %s\n%.4f" % (text[bi], ppl_last))
            print("[final complete sentence]: %s\n" % text_post[bi])

    return ppl_last, text, text_post


def counterfactual_reasoning(model, tokenizer, device, args, model_back=None):
    fr = open(args.input_file, 'r')
    data = [json.loads(x) for x in fr.readlines()]
    loss_rerank = 'norerank' if args.no_loss_rerank else 'rerank'
    file_name = '%s_%s_seed%d_%d_%d_%s_ngram%d_cw%.3f_lrnllp%.3f_len%d_topk%d_niter%d_frozlen%d' \
                '_winiter%d_noiseiter%d_gsstd%.4f_lr%.3f_%s_%s_output.json' % (
                    args.version,
                    loss_rerank,
                    args.seed,
                    args.start,
                    args.end,
                    args.mode,
                    args.counterfactual_max_ngram,
                    args.constraint_weight,
                    args.lr_nll_portion,
                    args.length,
                    args.topk,
                    args.num_iters,
                    args.frozen_length,
                    args.win_anneal_iters,
                    args.noise_iters,
                    args.gs_std,
                    args.stepsize,
                    args.large_noise_iters,
                    args.large_gs_std)

    outfile = os.path.join(args.output_dir, file_name)
    fw = open(outfile, 'w')
    fw_pretty = open(os.path.join(args.output_dir, 'pretty_' + file_name), 'w')
    fw_res = open(os.path.join(args.output_dir, 'res_' + file_name), 'w')

    procssed = set()
    for i, d in enumerate(data):
        if i < args.start or i > args.end:
            continue

        if args.seed != -1:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)

        print("%d / %d" % (i, len(data)))
        print('Output to: \t', outfile)
        print('output-lgt-temp:\t', args.output_lgt_temp)

        premise = d.get('premise', "")
        counterfactual = d.get('counterfactual', "")

        x = premise + ' ' + counterfactual
        ori_ending = d.get('original_ending', "")
        ori_endings = tokenize.sent_tokenize(ori_ending)

        if x in procssed:
            continue
        else:
            procssed.add(x)

        x_text_so_far = [""]
        x_addon = [[x]]

        outputs = []
        for oi, z_sent in enumerate(ori_endings):
            print("Sentence %d" % oi)
            z_text_so_far = z_sent.strip()
            z_text_so_far = ". " + z_text_so_far

            assert len(x_text_so_far) == len(x_addon), "%d vs %d" % (len(x_text_so_far), len(x_addon))

            new_x_text_so_far = []
            new_x_addon = []
            for ii, text_i in enumerate(x_text_so_far):
                for text_j in x_addon[ii]:
                    text_ij = text_i.strip() + " " + text_j.strip()
                    new_x_text_so_far.append(text_ij)

                    text_ij = text_ij.strip()

                    ppl_last, text, text_post = decode(
                        model, tokenizer, device, text_ij, z_text_so_far, None, args, model_back=model_back)

                    outputs.append([text_ij, text_post])

                    #  Rank and filter text_post from util.py
                    text_post = [post_sent(x) for x in text_post]
                    text_post = rank_and_filter(text_post, text_ij, z_text_so_far, model, tokenizer, device,
                                                args.no_loss_rerank)

                    if ii == len(x_text_so_far) - 1 and oi == len(ori_endings) - 1:
                        last_output = text_post
                        final_res = ' '.join([text_ij, last_output])
                        outputs.append(final_res)
                        fw_res.write(final_res + '\n')
                        fw_res.flush()

                    new_x_addon.append([text_post])

            x_text_so_far = new_x_text_so_far
            x_addon = new_x_addon

            break

        # complete_output = ' '.join(outputs)
        complete_output = outputs
        out = {
            'premise': premise,
            'initial': d.get('initial', ""),
            'counterfactual': counterfactual,
            'original_ending': ori_ending,
            'generation_complete': complete_output,
        }

        fw.write(json.dumps(out) + '\n')
        fw.flush()
        fw_pretty.write(json.dumps(out, indent=4) + '\n')
        fw_pretty.flush()

    print("outputs: %s" % outfile)


def grammar_correction(model, tokenizer, device, args, model_back=None):
    fr = open(args.input_file, 'r')
    data = [x.strip() for x in fr.readlines()]
    file_name = '%s_seed%d_%d_%d_%s_cw%.3f_lrnllp%.3f_len%d_topk%d_niter%d_frozlen%d' \
                '_winiter%d_noiseiter%d_gsstd%.4f_lr%.3f_%s_%s_output.json' % (
                    args.version,
                    args.seed,
                    args.start,
                    args.end,
                    args.mode,
                    args.constraint_weight,
                    args.lr_nll_portion,
                    args.length,
                    args.topk,
                    args.num_iters,
                    args.frozen_length,
                    args.win_anneal_iters,
                    args.noise_iters,
                    args.gs_std,
                    args.stepsize,
                    args.large_noise_iters,
                    args.large_gs_std)

    outfile = os.path.join(args.output_dir, file_name)
    fw = open(outfile, 'w')

    # Grammar
    data = [[' '.join(x.split()[:3]), ' '.join(x.split()[3:])] for x in data]

    print('#data: ', len(data))

    for i, d in enumerate(data):
        if i < args.start or i > args.end:
            continue
        print("%d / %d" % (i, len(data)))
        print('Output to: \t', outfile)

        if len(d[1].split()) <= 4:
            text = [d[1][2:]]
            text_post = [d[1][2:]]
            continue

        x = d[0]
        y = d[1]

        y = ". " + y

        ppl_last, text, text_post = decode(
            model, tokenizer, device, x, y, None, args, model_back=model_back)
        # break
        out = {
            'original': x + " " + y,
            'generation': text,
            'generation_complete': text_post,
        }

        fw.write(json.dumps(out) + '\n')

    print("outputs: %s" % outfile)


def abductive_reasoning(model, tokenizer, device, args, model_back=None):
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    stop_words = set(stopwords.words('english'))

    def _get_adverbs_and_nnps(z_words):
        pos = nltk.pos_tag(z_words)
        adverbs = [w[0] for w in pos if 'RB' in w[1]]
        nnps = [w[0] for w in pos if 'NNP' in w[1]]
        return adverbs, nnps

    def _get_keywords(z, x):
        z_words = word_tokenize(z)
        z_adverbs, z_nnps = _get_adverbs_and_nnps(z_words)
        ret_words = []
        for w in z_words:
            if w in z_nnps:
                if w not in ret_words:
                    ret_words.append(w)
            else:
                w = w.lower()
                if w not in stop_words and w.isalnum() and w not in z_adverbs and w not in ret_words:
                    ret_words.append(w)

        if args.abductive_filterx:
            x_words = word_tokenize(x)
            ret_words = [w for w in ret_words if w not in x_words]

        return ' '.join(ret_words)

    with open(args.input_file, 'r') as f:
        lines = f.readlines()
        data = [json.loads(l.strip()) for l in lines]

    outfile = '%s_seed%d_%d_%d_%s_cw%.3f_c2w%.3f_lrnllp%.3f_len%d_topk%d_niter%d_frozlen%d' \
              '_winiter%d_noiseiter%d_gsstd%.4f_lr%.3f_lrratio%.2f_lriter%d_%s_%s_output.json' % (
                  args.version,
                  args.seed,
                  args.start,
                  args.end,
                  args.mode,
                  args.constraint_weight,
                  args.abductive_c2_weight,
                  args.lr_nll_portion,
                  args.length,
                  args.topk,
                  args.num_iters,
                  args.frozen_length,
                  args.win_anneal_iters,
                  args.noise_iters,
                  args.gs_std,
                  args.stepsize,
                  args.stepsize_ratio,
                  args.stepsize_iters,
                  args.large_noise_iters,
                  args.large_gs_std)
    print("outputs: %s" % outfile)

    fw = open(os.path.join(args.output_dir, outfile), 'w')
    fw_pretty = open(os.path.join(args.output_dir, 'pretty_' + outfile), 'w')

    procssed = set()
    for i, d in enumerate(data):
        if i < args.start or i > args.end:
            continue

        x = d["obs1"].strip()
        #x = d["obs2"].strip() + '<|endoftext|>' + d["obs1"].strip()
        z = d["obs2"].strip()
        z_keywords = _get_keywords(z, x)

        if ' '.join([x, z]) in procssed:
            continue
        procssed.add(' '.join([x, z]))

        print("%d / %d" % (i, len(data)))
        print('Output to: \t', outfile)

        z = ". " + z
        z_keywords = ". " + z_keywords
        ppl_last, text, text_post = decode(model, tokenizer, device, x, z, None, args, model_back=model_back,
                                           zz=z_keywords)

        out = {
            'x': x,
            'z': z,
            'z_keywords': z_keywords,
            'ppl_last': float(ppl_last),
            'generation': text,
            'generation_complete': text_post,
        }

        fw.write(json.dumps(out) + '\n')
        fw.flush()
        fw_pretty.write(json.dumps(out, indent=4) + '\n')
        fw_pretty.flush()

    print("outputs: %s" % outfile)


def lexical_generation(model, tokenizer, device, args, model_back=None):
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    stop_words = set(stopwords.words('english'))

    def _get_adverbs_and_nnps(z_words):
        pos = nltk.pos_tag(z_words)
        adverbs = [w[0] for w in pos if 'RB' in w[1]]
        nnps = [w[0] for w in pos if 'NNP' in w[1]]
        return adverbs, nnps

    with open(args.input_file, 'r') as f:
        lines = f.readlines()
        data = [json.loads(l.strip()) for l in lines]

    outfile = '%s_seed%d_%d_%d_%s_cw%.3f_c2w%.3f_lrnllp%.3f_len%d_topk%d_niter%d_frozlen%d' \
              '_winiter%d_noiseiter%d_gsstd%.4f_lr%.3f_lrratio%.2f_lriter%d_%s_%s_output.json' % (
                  args.version,
                  args.seed,
                  args.start,
                  args.end,
                  args.mode,
                  args.constraint_weight,
                  args.abductive_c2_weight,
                  args.lr_nll_portion,
                  args.length,
                  args.topk,
                  args.num_iters,
                  args.frozen_length,
                  args.win_anneal_iters,
                  args.noise_iters,
                  args.gs_std,
                  args.stepsize,
                  args.stepsize_ratio,
                  args.stepsize_iters,
                  args.large_noise_iters,
                  args.large_gs_std)
    print("outputs: %s" % outfile)

    fw = open(os.path.join(args.output_dir, outfile), 'w')
    fw_pretty = open(os.path.join(args.output_dir, 'pretty_' + outfile), 'w')

    procssed = set()
    for i, d in enumerate(data):
        if i < args.start or i > args.end:
            continue
        print(d)
        constraints = d["concept_set"].split("#")

        constraints = ' '.join(constraints)
        x = "<|endoftext|>"
        z = constraints
        z_keywords = constraints

        print("%d / %d" % (i, len(data)))
        print('Output to: \t', outfile)

        z = ". " + z
        z_keywords = ". " + z_keywords

        text_candidates = []
        text_complete_candidates = []
        for _ in range(int(args.batch_size / 32)):
            ppl_last, text, text_post = decode(model, tokenizer, device, x, z, None, args, model_back=model_back,
                                               zz=z_keywords)
            text_candidates.extend(text)
            text_complete_candidates.extend(text_post)

        out = {
            'x': x,
            'constraints': constraints,
            'generation': text_candidates,
            'generation_complete': text_complete_candidates,
        }
        print(out)
        print('Output to: \t', outfile)

        fw.write(json.dumps(out) + '\n')
        fw.flush()
        fw_pretty.write(json.dumps(out, indent=4) + '\n')
        fw_pretty.flush()

    print("outputs: %s" % outfile)


def main():
    args = options()
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    if args.seed != -1:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    # Load pretrained model
    model = GPT2LMHeadModel.from_pretrained(
        args.pretrained_model, output_hidden_states=True,
        resid_pdrop=0, embd_pdrop=0, attn_pdrop=0, summary_first_dropout=0)
    model.to(device)
    model.eval()
    # Freeze GPT-2 weights
    for param in model.parameters():
        param.requires_grad = False

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.pretrained_model)

    model_back = OpenGPT2LMHeadModel.from_pretrained(
        args.back_model, hidden_dropout_prob=0, attention_probs_dropout_prob=0, summary_first_dropout=0)
    model_back.to(device)
    model_back.eval()
    # Freeze GPT-2 weights
    for param in model_back.parameters():
        param.requires_grad = False

    if "lexical" in args.mode:
        lexical_generation(model, tokenizer, device, args, model_back)
    if "counterfactual" in args.mode:
        counterfactual_reasoning(model, tokenizer, device, args, model_back)
    if "abductive" in args.mode:
        abductive_reasoning(model, tokenizer, device, args, model_back)
    if "grammar" in args.mode:
        grammar_correction(model, tokenizer, device, args, model_back)


if __name__ == "__main__":
    main()
