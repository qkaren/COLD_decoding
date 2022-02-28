import torch
import torch.nn.functional as F
import json
import os
import nltk
from nltk import tokenize
import torch
import numpy as np

nltk.download('punkt')

import sys
import os
if os.path.isdir('/var/karen'):
    os.environ['TRANSFORMERS_CACHE'] = '/var/karen/workspace/Refinement-Generation/cache'
    sys.path.insert(0, '/var/karen/workspace/Refinement-Generation/')

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm
from difflib import SequenceMatcher

from bleuloss import batch_log_bleulosscnn_ae
from util import *


def embed_inputs(embedding, logits, x_onehot=None, z_onehot=None, device='cuda'):
    '''
    embeds inputs in a dense representation, before passing them to the model
    '''
    # typically we embed a one-hot vector. But here since we work we work with dense representations,
    # we have softmax here to make sure that all the values of the input logits sum to one (similar to a 1-hot vector).
    probs = F.softmax(logits, dim=-1)

    if x_onehot is not None:
        probs = torch.cat((x_onehot.type(torch.FloatTensor), probs.type(torch.FloatTensor)), dim=1)
    if z_onehot is not None:
        probs = torch.cat((probs.type(torch.FloatTensor), z_onehot.type(torch.FloatTensor)), dim=1)
    probs = probs.to(device)

    return torch.matmul(probs, embedding)


def _greedy(logits):
    _, last = torch.topk(logits, k=1, dim=-1)
    return last


def top_k_filter_3d(logits, k, probs=False, mask=None, extra_mask=None):
    """
    logits.shape = [batch_size, length, vocab_size]
    extra_mask: [batch_size, length, vocab_size], 1 if reserve
    """
    BIG_CONST = 1e10
    if k == 0:
        return logits
    else:
        if mask is None:
            _, indices = torch.topk(logits, k)
            mask = torch.zeros_like(logits).scatter_(2, indices, 1)
        if extra_mask is not None:
            mask = ((mask + extra_mask) > 0).float()
        if probs:
            return logits * mask
        return logits * mask + -BIG_CONST * (1-mask)


def top_k_filter(logits, k, probs=False):
    BIG_CONST = 1e10
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        if probs:
            return torch.where(logits < batch_mins, torch.ones_like(logits) * 0.0, logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -BIG_CONST, logits)


def _topk(logits, k=10):
    logits = top_k_filter(logits, k)
    probs = F.softmax(logits, dim=-1)
    last = torch.multinomial(probs, num_samples=1)
    return last


def get_text_from_logits(logits, tokenizer):
    output_so_far = None
    last = None
    logp = 0
    for i in range(logits.shape[1]):
        last = _greedy(logits[:, i, :])
        output_so_far = last if output_so_far is None else torch.cat((output_so_far, last), dim=1)
        logp += logits[:, i, :].log_softmax(-1).data.cpu().numpy()[:, last.data.cpu().numpy()]

    nll = -logp
    batch_size = output_so_far.shape[0]
    text = []
    for i in range(batch_size):
        text_i = tokenizer.decode(output_so_far[i].tolist())
        text_i = text_i.replace('\n', ' ')
        text.append(text_i)

    return text, nll, output_so_far


def get_text_from_logits_topk(logits, tokenizer, top_k=1):
    output_so_far = None
    last = None
    logp = 0

    for i in range(logits.shape[1]):
        last = _topk(logits[:, i, :], top_k)
        output_so_far = last if output_so_far is None else torch.cat((output_so_far, last), dim=1)
        logp += logits[:, i, :].log_softmax(-1)[:, last.item()].item()

    nll = -logp
    text = tokenizer.decode(output_so_far.tolist()[0])
    text = text.replace('\n', ' ')
    return text, nll, output_so_far


def one_hot(tensor, dimension):
    while len(tensor.shape) < 2:
        tensor = tensor.unsqueeze(0)
    onehot = torch.LongTensor(tensor.shape[0], tensor.shape[1], dimension).to(tensor.device)
    onehot.zero_().scatter_(2, tensor.unsqueeze(-1), 1)
    onehot.to(tensor.device)
    return onehot


def initialize(model, x, length, temperature, device):
    if x.dim() == 1:
        x = x.unsqueeze(0)
    past = None
    last_token_embedding = None
    logits_so_far = None
    for i in range(length):
        # for the first iteration, `past` is None
        if past is None:
            x_last_token = x[:, -1:]
            last_token_embedding = model.get_input_embeddings()(x_last_token)

            # if the input length is longer than a single token
            if x.shape[1] > 1:
                x_except_last_token = x[:, :-1]
                model_outputs = model(x_except_last_token)
                past = model_outputs.past_key_values

        model_outputs = model(past_key_values=past, inputs_embeds=last_token_embedding)
        logits = model_outputs.logits
        past = model_outputs.past_key_values

        logits = logits[:, -1, :] / temperature
        logits = logits.unsqueeze(1)
        logits_so_far = logits if logits_so_far is None else torch.cat((logits_so_far, logits), dim=1)
        last_token_embedding = embed_inputs(embedding=model.get_input_embeddings().weight, logits=logits, device=device)

    return logits_so_far


def decode_with_model_topk(model, y_logits, topk, x_onehot, x_past, tokenizer, extra_mask=None):
    assert x_onehot.shape[1] == 1, x_onehot.shape
    length = y_logits.shape[1]
    past = x_past
    input_embeds = torch.matmul(x_onehot.float(), model.get_input_embeddings().weight)
    mask_t_all = None
    logits_so_far = None
    for i in range(length):
        model_outputs = model(past_key_values=past, inputs_embeds=input_embeds)
        past = model_outputs.past_key_values
        logits_t = model_outputs.logits[:, -1:, :]
        assert logits_t.shape[1] == 1, logits_t.shape
        _, indices_t = torch.topk(logits_t, topk)
        mask_t = torch.zeros_like(logits_t).scatter_(2, indices_t, 1)
        mask_t_all = mask_t if mask_t_all is None else torch.cat((mask_t_all, mask_t), dim=1)
        logits_so_far = logits_t if logits_so_far is None else torch.cat((logits_so_far, logits_t), dim=1)
        if i < length - 1:
            if extra_mask is None:
                y_logits_i_topk = top_k_filter_3d(y_logits[:,i:i+1,:], topk, mask=mask_t) / 0.001
            else:
                y_logits_i_topk = top_k_filter_3d(y_logits[:,i:i+1,:], topk, mask=mask_t, extra_mask=extra_mask[:,i:i+1,:]) / 0.001
            input_embeds = torch.matmul(F.softmax(y_logits_i_topk, dim=-1), model.get_input_embeddings().weight)
    return get_text_from_logits(
        top_k_filter_3d(y_logits, topk, mask=mask_t_all, extra_mask=extra_mask),
        tokenizer)


def post_process(text_ids, model, max_length, length, tokenizer, device):
    # sentence completion
    text_ids_complete = sentence_completion(text_ids, model, max_length, device)
    batch_size = text_ids.shape[0]
    text_so_far_all = []
    for bi in range(batch_size):
        text_complete = tokenizer.decode(text_ids_complete[bi].tolist())
        text_complete = text_complete.replace('\n', ' ')

        # truncate to minimal complete text
        sents = nltk.sent_tokenize(text_complete)
        text_so_far = None
        length_so_far = 0
        for i, sent in enumerate(sents):
            text_so_far = sent if text_so_far is None else text_so_far + ' ' + sent
            sent_length = len(sent.split())
            length_so_far += sent_length
            if length_so_far >= length:
                break
        text_so_far_all.append(text_so_far)
    return text_so_far_all


def sentence_completion(text_ids, model, max_length, device):
    output_so_far = text_ids
    past = None
    last_embeds = None
    # logits_so_far = None
    for i in range(max_length - text_ids.shape[1]):
        if past is None and output_so_far is not None:
            last = output_so_far[:, -1:]
            last_embeds = model.get_input_embeddings()(last)

            if output_so_far.shape[1] > 1:
                model_outputs = model(output_so_far[:, :-1])
                past = model_outputs.past_key_values

        model_outputs = model(past_key_values=past, inputs_embeds=last_embeds)
        logits = model_outputs.logits
        past = model_outputs.past_key_values

        last = _greedy(logits[:, -1, :])
        output_so_far = last if output_so_far is None else torch.cat((output_so_far, last), dim=1)
        # last_embeds = get_input_embeds(model.get_input_embeddings(), logits[:, -1:, :], device=device)
        last_embeds = model.get_input_embeddings()(last)

    return output_so_far


def soft_distance(logits_perturbed, logits):
    return torch.nn.MSELoss()(logits_perturbed, logits)


def soft_nll(logits_perturbed, logits):
    p = F.softmax(logits_perturbed, dim=-1)
    logp = F.log_softmax(logits, dim=-1)
    return -(p * logp).sum(dim=-1).mean(dim=-1)


def soft_nll_detach(logits_perturbed, logits):
    p = F.softmax(logits_perturbed, dim=-1).detach()
    logp = F.log_softmax(logits, dim=-1)
    return -(p * logp).sum(dim=-1).mean()


def additional_nll(logits, cur_text_ids):
    return torch.nn.CrossEntropyLoss()(
        logits.view(-1, logits.shape[-1]),
        cur_text_ids.view(-1)
    )


def soft_forward(model, x_onehot, y_logits, x_past=None, detach=True):
    '''
    computes logits for $y$, based on a fixed context $y$ and the current logit distribution of $y$
    :param model:
    :param x_onehot:
    :param y_logits:
    :return:
    '''
    xy_embeds = embed_inputs(
        model.get_input_embeddings().weight,
        y_logits,
        x_onehot=x_onehot,
        device=x_onehot.device
    )
    xy_logits = model(past_key_values=x_past, inputs_embeds=xy_embeds).logits
    x_length = x_onehot.shape[1]
    y_logits = xy_logits[:, x_length - 1:-1, :]
    if detach:
        return y_logits.detach()
    else:
        return y_logits


def soft_forward_xyz(model, x_onehot, y_logits, z_onehot):
    '''
    computes logits for $y$, based on a fixed context $y$ and the current logit distribution of $y$
    :param model:
    :param x_onehot:
    :param y_logits:
    :return:
    '''
    xyz_embeds = embed_inputs(
        model.get_input_embeddings().weight,
        y_logits,
        x_onehot=x_onehot,
        z_onehot=z_onehot,
        device=y_logits.device
    )
    xyz_logits = model(inputs_embeds=xyz_embeds).logits
    if x_onehot is not None:
        xy_length = x_onehot.shape[1] + y_logits.shape[1]
    else:
        xy_length = y_logits.shape[1]
    return xyz_logits, xy_length



def soft_backward(model, y_logits_rev):
    embeddings_weight = model.get_input_embeddings().weight[1:y_logits_rev.shape[-1]+1]
    y_embeds = embed_inputs(
        embeddings_weight,
        y_logits_rev,
        device=y_logits_rev.device
    )
    y_logits_ = model(inputs_embeds=y_embeds).logits
    return y_logits_[:, :-1, :]


def soft_backward_steps(model, y_logits):
    device = y_logits.device
    past = None
    last_embeds = None
    logits_so_far = None
    for i in range(y_logits.shape[1]-2, -1, -1):
        last = y_logits[:, i:i+1]
        last_embeds = embed_inputs(model.get_input_embeddings(), last, device=device)

        model_outputs = model(past_key_values=past, inputs_embeds=last_embeds)
        past = model_outputs.past_key_values

        logits = model_outputs.logits
        logits = logits[:, -1, :]
        logits = logits.unsqueeze(1)
        logits_so_far = logits if logits_so_far is None else torch.cat((logits_so_far, logits), dim=1)

    return logits_so_far



def constraint_loss(logits, cs_onehot, cs_ids):
    """
    constraint loss with mask
    cs_ids: [batch_size, num_cs]
    """
    log_ps = logits.log_softmax(-1).unsqueeze(2)  # shape: [batch_size, length, 1, vocab_size]
    constraint_max_log_ps_ = (log_ps * cs_onehot.unsqueeze(1)).max(1)[0].sum(-1)  # shape: [batch_size, num_cs]

    log_ps_max_ids = log_ps[:, :, 0, :].argmax(-1)  # shape: [batch_size, length]
    cs_ids_repeat = cs_ids.unsqueeze(2).repeat([1, 1, log_ps_max_ids.shape[1]])  # shape: [batch_size, num_cs, length]
    mask = (log_ps_max_ids.unsqueeze(1) == cs_ids_repeat).type(torch.FloatTensor).sum(-1)  # shape: [batch_size, num_cs]
    mask = (mask < 1).type(torch.FloatTensor)
    mask = mask.to(constraint_max_log_ps_.device)

    loss = - (constraint_max_log_ps_ * mask).sum()

    if mask.sum() != 0:
        loss = loss / mask.sum()
    else:
        loss = 0

    return loss


def constraint_loss_with_variants(logits, cs_onehot_all, cs_ids_all):
    """
    constraint loss with mask
    cs_ids_all: list of tensor [batch_size, num_variants], of length num_cs
    """
    device = logits.device
    log_ps = logits.log_softmax(-1).unsqueeze(2)  # shape: [batch_size, length, 1, vocab_size]

    num_cs = len(cs_onehot_all)
    loss_all = 0
    mask_sum = 0
    for i in range(num_cs):
        cs_onehot = cs_onehot_all[i]
        cs_ids = cs_ids_all[i]
        constraint_max_log_ps_ = (log_ps * cs_onehot.unsqueeze(1)).max(1)[0].sum(-1)  # shape: [batch_size, num_variants]

        log_ps_max_ids = log_ps[:, :, 0, :].argmax(-1)  # shape: [batch_size, length]
        cs_ids_repeat = cs_ids.unsqueeze(2).repeat([1, 1, log_ps_max_ids.shape[1]])  # shape: [batch_size, num_variants, length]
        mask = (log_ps_max_ids.unsqueeze(1) == cs_ids_repeat).type(torch.FloatTensor).sum(-1)  # shape: [batch_size, num_variants]
        #mask = (mask >= 1).type(torch.FloatTensor)
        mask = (mask.sum(1) < 1).type(torch.FloatTensor)  # shape: [batch_size]. mask = 0 if any of the variants already occurs
        mask = mask.to(device)

        loss_i = - (constraint_max_log_ps_.max(1)[0] * mask).mean()  # average over batch_size

        loss_all += loss_i
        mask_sum += mask

    if mask_sum != 0:
        loss_all = loss_all / mask_sum

    return loss_all #, mask_sum


def constraint_loss_with_variants_by_ppl(logits, cs_onehot_all, cs_ids_all, probs_t):
    device = logits.device
    batch_size = logits.shape[0]
    log_ps = logits.log_softmax(-1).unsqueeze(2)
    ps_t = probs_t.unsqueeze(2)

    num_cs = len(cs_onehot_all)
    loss_all = 0
    mask_sum = 0
    for i in range(num_cs):
        cs_onehot = cs_onehot_all[i]
        cs_ids = cs_ids_all[i]

        cs_onehot_ = cs_onehot.unsqueeze(1).type(torch.FloatTensor).to(device)
        cs_onehot_ = cs_onehot_.repeat(batch_size, 1, 1, 1).type(torch.FloatTensor).to(device)
        ppl_max_idx = (ps_t * cs_onehot_).argmax(1)  # [batch_size, num_variants, vocab_size]
        ppl_max_idx_onehot = torch.zeros_like(log_ps * cs_onehot_).scatter_(1, ppl_max_idx.unsqueeze(1), cs_onehot_)

        constraint_max_log_ps_ = (log_ps * ppl_max_idx_onehot).sum(1).sum(-1)  # shape: [batch_size, num_variants]

        ## Mask
        log_ps_max_ids = log_ps[:, :, 0, :].argmax(-1)  # shape: [batch_size, length]
        cs_ids_repeat = cs_ids.unsqueeze(2).repeat([1, 1, log_ps_max_ids.shape[1]])  # shape: [batch_size, num_variants, length]
        mask = (log_ps_max_ids.unsqueeze(1) == cs_ids_repeat).type(torch.FloatTensor).sum(-1)  # shape: [batch_size, num_variants]
        mask = (mask.sum(1) < 1).type(torch.FloatTensor)  # shape: [batch_size]. mask = 0 if any of the variants already occurs
        mask = mask.to(device)

        loss_i = - constraint_max_log_ps_.max(1)[0] * mask

        loss_all += loss_i  # shape: [batch_size]
        mask_sum += mask  # shape: [batch_size]

    loss_all = loss_all / (mask_sum + 1e-8)

    return loss_all


def constraint_loss_by_ppl(logits, cs_onehot, cs_ids, logits_t):
    device = logits.device
    log_ps = logits.log_softmax(-1).unsqueeze(2)

    cs_onehot_ = cs_onehot.unsqueeze(1).type(torch.FloatTensor).to(device)
    ps_t = logits_t.softmax(-1).unsqueeze(2)
    ppl_max_idx = (ps_t * cs_onehot_).argmax(1)  # [batch_size, num_cs, vocab_size]
    ppl_max_idx_onehot = torch.zeros_like(log_ps * cs_onehot_).scatter_(1, ppl_max_idx.unsqueeze(1), cs_onehot_)

    constraint_max_log_ps_ = (log_ps * ppl_max_idx_onehot).sum(1).sum(-1)  # shape: [batch_size, num_cs]

    ## Mask
    log_ps_max_ids = log_ps[:, :, 0, :].argmax(-1)  # shape: [batch_size, length]
    cs_ids_repeat = cs_ids.unsqueeze(2).repeat([1, 1, log_ps_max_ids.shape[1]])  # shape: [batch_size, num_cs, length]
    mask = (log_ps_max_ids.unsqueeze(1) == cs_ids_repeat).type(torch.FloatTensor).sum(-1)  # shape: [batch_size, num_cs]
    mask = (mask < 1).type(torch.FloatTensor)
    mask = mask.to(device)

    loss = - (constraint_max_log_ps_ * mask).sum()

    if mask.sum() != 0:
        loss = loss / mask.sum()
    else:
        loss = 0

    return loss


def constraint_loss_all(logits, cs_onehot, cs_ids):
    device = logits.device

    log_ps = logits.log_softmax(-1).unsqueeze(2)
    constraint_max_log_ps_ = (log_ps * cs_onehot.unsqueeze(1)).mean(1).sum(-1)  # shape: [batch_size, num_cs]

    ## Mask
    log_ps_max_ids = log_ps[:, :, 0, :].argmax(-1)  # shape: [batch_size, length]
    cs_ids_repeat = cs_ids.unsqueeze(2).repeat([1, 1, log_ps_max_ids.shape[1]])  # shape: [batch_size, num_cs, length]
    mask = (log_ps_max_ids.unsqueeze(1) == cs_ids_repeat).type(torch.FloatTensor).sum(-1)  # shape: [batch_size, num_cs]
    mask = (mask < 1).type(torch.FloatTensor)
    mask = mask.to(device)

    loss = - (constraint_max_log_ps_ * mask).sum()

    if mask.sum() != 0:
        loss = loss / mask.sum()
    else:
        loss = 0

    return loss

def _constraint_loss2(logits, cs_onehot):
    '''
    a re-implementation of `_constraint_loss` with a slightly different logic.
    TODO: keep only one of these functions
    '''
    logits = logits.squeeze(0) # drop the empty dimension
    cs_onehot = cs_onehot.float().squeeze(0) # drop the empty dimension and change into float (since torch matrix multiplication does not support integers)
    cs_onehot = torch.transpose(cs_onehot, 0, 1)
    selected_logits = torch.matmul(logits, cs_onehot) # dim: length x # of constraints
    max_logits_per_constraint, _ = selected_logits.max(0) # select the highest logits for each constraint
    loss = - max_logits_per_constraint.sum() / selected_logits.size(1)
    return loss

def print_topk_stats(logits, tokenizer):
    logits_lg, topk_index_y = torch.topk(F.softmax(logits[0, :3, :], dim=-1), 3)
    print(logits_lg.data.cpu().numpy())
    print(topk_index_y.data.cpu().numpy())
    lgs = [int(x[0]) for x in topk_index_y.data.cpu().numpy()]
    for a in lgs:
        print('|', tokenizer.decode(a), '| ', end='', flush=True)
    print()
    print("===============================")
    return topk_index_y



def collect_json_lines(model_output_json_file):
    with open(model_output_json_file, 'r') as fr:
        lines = fr.readlines()
        json_lines = [json.loads(x.strip()) for x in lines]
        return json_lines

def post_sent(text_complete):
    sents = nltk.sent_tokenize(text_complete)
    sent = ' '.join(sents[0].strip().split())
    return sent
    # return sents[0]

def _has_repeat_sent(hyp):
    """
    Detect if the sentences in `hyp` are repeat.
    Args:
        hyp: A list of three sentences.
    """
    if len(hyp) <= 1:
        return False

    for i in range(1, len(hyp)):
        a = hyp[i-1]
        b = hyp[i]

        if a == b:
            return True

        s = SequenceMatcher(None, a, b)
        if len(a) > 5 and len(b) > 5 and s.ratio() >= 0.85:
            return True

    return False


def _has_repeat_substring(s, MINLEN=4, MINCNT=4):
    d = {}
    has_repeat = False
    for sublen in range(int(len(s)/MINCNT)-1, MINLEN-1, -1):
        for i in range(0, len(s)-sublen):
            sub = s[i:i+sublen]
            if len(sub.strip()) < sublen:
                continue
            cnt = s.count(sub)
            if cnt >= MINCNT and sub not in d:
                 d[sub] = cnt
                 print('repeat_substring: |' + sub + '| in |' + s + '|')
                 has_repeat = True
                 break
        if has_repeat:
            break
    return has_repeat


def has_repeat(sents_for_substr, sents_for_sent):
    """
    Detect if the hypothesis text has repeat patterns.
    """
    has_repeat_substring = False
    for h in sents_for_substr:
        has_repeat_substring = has_repeat_substring or _has_repeat_substring(h) or _has_repeat_substring(h, MINLEN=20, MINCNT=2)
    # print(has_repeat_substring)
    # print(_has_repeat_sent(hyp))
    return has_repeat_substring or _has_repeat_sent(sents_for_sent)


def write_json_lines(json_lines, fout, model, tokenizer, device):
    with open(fout, 'w') as fw:
        for line in json_lines:
            input_text = line['generation_complete'][0][0]
            # input_text = line['counterfactual']

            ori_ending = line['original_ending']
            ori_endings = tokenize.sent_tokenize(ori_ending)
            z = ori_endings[0].strip()

            gens = line['generation_complete'][0][1]
            proc_gens = [post_sent(x) for x in gens]
            pg_dict, gens_ranked, pg_dict_top, gens_ranked_top = process_batching_counterfactual_outputs(
                proc_gens, input_text, z, model, tokenizer, device)
            line['proced'] = proc_gens
            line['ppl_gens'] = pg_dict
            line['gens_ranked'] = gens_ranked
            line['ppl_gens_top'] = pg_dict_top
            line['gens_ranked_top'] = gens_ranked_top
            # print(line)
            # exit()
            fw.write(json.dumps(line) + '\n')


def compute_ppl_line(model, tokenizer, device, line):
    line = line.strip()
    #print(line)
    line_ = tokenizer.encode(line)
    line_t = torch.tensor(line_, device=device, dtype=torch.long)
    loss = model(input_ids=line_t, labels=line_t).loss
    loss = loss.detach().clone().data.cpu().numpy()
    ppl = np.exp(loss)
    return ppl


def compute_loss(model, tokenizer, device, x="", z="", y="", constraints=None, args=None, model_back=None, zz=None):
    '''
    x: left context   (prompt in lexical constrained task)
    z: optimization target  (original ending in counterfactual task)
    constraints: (constraint set in lexical constrained task)
    '''
    batch_size = 2

    x_ = tokenizer.encode(x)
    x_t = torch.tensor(x_, device=device, dtype=torch.long)
    x_onehot = one_hot(x_t, dimension=tokenizer.vocab_size)

    # repeat batch_size times
    x_t = x_t.unsqueeze(0).repeat(batch_size, 1)
    x_onehot = x_onehot.repeat(batch_size, 1, 1)

    z_ = tokenizer.encode(z)[1:] # delete the "." token we appended before
    z_t = torch.tensor(z_, device=device, dtype=torch.long)
    z_t = z_t.unsqueeze(0).repeat(batch_size, 1)

    y_ = tokenizer.encode(y)[1:] # delete the "." token we appended before
    y_t = torch.tensor(y_, device=device, dtype=torch.long)
    y_onehot = one_hot(y_t, dimension=tokenizer.vocab_size)
    y_onehot = y_onehot.repeat(batch_size, 1, 1)
    y_t = y_t.unsqueeze(0).repeat(batch_size, 1)

    y_logits_ = y_onehot / 0.0001

    c_loss = batch_log_bleulosscnn_ae(
        decoder_outputs=y_logits_.transpose(0, 1),
        target_idx=z_t,
        ngram_list=[2, 3]
    )

    return c_loss.mean().item()


def rank_and_filter(candidates, input_text, z, model, tokenizer, device, no_loss_rerank):

    # de-duplicate
    candidates = list(dict.fromkeys(candidates))

    ppl_list = []
    ppl_y_list = []
    loss_list = []
    for line in candidates:
        line = line.strip()
        y = ' '.join(line.split())
        # y = line
        xy = input_text + ' ' + line
        # print(xy)
        # exit()
        x_sents = nltk.sent_tokenize(input_text)
        if has_repeat(sents_for_substr=[y], sents_for_sent=x_sents+[y]) or len(tokenizer.encode(y)) <= 4:
            ppl_list.append(10000.0)
            ppl_y_list.append(10000.0)
            loss_list.append(10000.0)
        else:
            ppl = compute_ppl_line(model, tokenizer, device, xy)
            ppl_list.append(round(ppl, 2))

            ppl_y = compute_ppl_line(model, tokenizer, device, y)
            ppl_y_list.append(round(ppl_y, 2))

            loss = compute_loss(model, tokenizer, device,
                                x=input_text, z=". " + z, y=". " + y)
            loss_list.append(loss)

    sort_index = sorted(range(len(ppl_list)), key=lambda k: ppl_list[k])
    ppls_reorder = [ppl_list[i] for i in sort_index]
    ppls_y_reorder = [ppl_y_list[i] for i in sort_index]
    loss_reorder = [loss_list[i] for i in sort_index]
    gens_complete_reorder = [candidates[i] for i in sort_index]

    pg_dict = []
    for p, py, l, g in zip(ppls_reorder, ppls_y_reorder, loss_reorder, gens_complete_reorder):
        pg_dict.append({"ppl": str(p), "ppl_y": str(py), "loss": str(l), "gen": g})

    if len(ppls_reorder) <= 1:
        sort_len = 1
    elif ppls_reorder[1]-ppls_reorder[0] > 10:
        sort_len = 1
    elif len(ppls_reorder) <= 2:
        sort_len = 1
    elif ppls_reorder[2]-ppls_reorder[0] > 10:
        sort_len = 2
    else:
        sort_len = 3

    if no_loss_rerank:
        return gens_complete_reorder[0]

    sort_index = sorted(range(sort_len), key=lambda k: loss_reorder[k])
    sort_index = sort_index
    ppls_reorder_top = [ppls_reorder[i] for i in sort_index]
    ppls_y_reorder_top = [ppls_y_reorder[i] for i in sort_index]
    loss_reorder_top = [loss_reorder[i] for i in sort_index]
    gens_complete_reorder_top = [gens_complete_reorder[i] for i in sort_index]

    pg_dict_top = []
    for p, py, l, g in zip(ppls_reorder_top, ppls_y_reorder_top, loss_reorder_top, gens_complete_reorder_top):
        pg_dict_top.append({"ppl": str(p), "ppl_y": str(py), "loss": str(l), "gen": g})

    return gens_complete_reorder_top[0]


