from torch.cuda import LongTensor, FloatTensor
import torch
from torch import nn
import torch.nn.functional as F
# # decoder_outputs: length x batch x vocab
# # target: length x batch x vocab
# # (decoder_outputs, target_variable, batch_size, sentence_length, if_input=False, if_target=False)
# def batch_log_bleulosscnn(decoder_outputs, target_variable, batch_size, sentence_length, if_input=False, if_target=False, pad=0, weight_list=None, ngram_list=None):
#     """
#     decoder_outputs - matrix with probabilityes  not convert to prob domain   l x b x v
#     target_variable - reference batch -- prob       1 x b x v
#     maxorder - max order of n-gram
#     translation_lengths -  lengths of the translations - torch tensor
#     reference_lengths - lengths of the references - torch tensor
#     """
#     if ngram_list[0] <= 0:
#         ngram_list[0] = sentence_length
#     if type(ngram_list) == int:
#         ngram_list = [ngram_list]
#     if weight_list is None:
#         weight_list = [1./len(ngram_list)] * len(ngram_list)
#     weight_list = [0.8,0.2]
#     decoder_outputs = torch.log(torch.softmax(decoder_outputs, dim=-1)) # log domain
#     pred_onehot = torch.cat(decoder_outputs.transpose_(0,1).contiguous().chunk(batch_size, 0), -1).transpose(1, 2).unsqueeze(-2)
#     target_onehot = torch.cat(target_variable.chunk(batch_size, 1), 0).transpose(1,2).unsqueeze(-1)
#     out = nn.functional.conv2d(pred_onehot, target_onehot, groups=batch_size)
#     out = torch.cat(out.chunk(batch_size, 1), 0).permute(0, 2, 3, 1)
#     sum_gram = FloatTensor([0.])

#     for cnt, ngram in enumerate(ngram_list):
#         eye_filter = torch.eye(ngram).view([1, 1, ngram, ngram]).cuda()
#         term = nn.functional.conv2d(out, eye_filter) / ngram
#         if ngram < decoder_outputs.size()[1]:
#             gum_tmp = F.gumbel_softmax(term.squeeze_(1), tau=1, dim=2)
#             term = term.mul(gum_tmp).sum(1).mean(1)
#         sum_gram += weight_list[cnt]*term.sum()
#     loss = - sum_gram.sum() / batch_size
#     return loss



def batch_log_bleulosscnn_ae(decoder_outputs, target_idx, ngram_list, trans_len=None, pad=0, weight_list=None):
    """
    decoder_outputs: [output_len, batch_size, vocab_size]
        - matrix with probabilityes  -- log probs
    target_variable: [batch_size, target_len]
        - reference batch
    ngram_list: int or List[int]
        - n-gram to consider
    pad: int
        the idx of "pad" token
    weight_list : List
        corresponding weight of ngram

    NOTE: output_len == target_len
    """
    decoder_outputs = decoder_outputs.transpose(0,1)
    batch_size, output_len, vocab_size = decoder_outputs.size()
    _, tgt_len = target_idx.size()
    if type(ngram_list) == int:
        ngram_list = [ngram_list]
    if ngram_list[0] <= 0:
        ngram_list[0] = output_len
    if weight_list is None:
        weight_list = [1. / len(ngram_list)] * len(ngram_list)
    decoder_outputs = torch.log_softmax(decoder_outputs,dim=-1)
    decoder_outputs = torch.relu(decoder_outputs + 20) - 20
    index = target_idx.unsqueeze(1).expand(-1, output_len, tgt_len)
    cost_nll = decoder_outputs.gather(dim=2, index=index)
    cost_nll = cost_nll.unsqueeze(1)
    out = cost_nll
    sum_gram = 0. #FloatTensor([0.])
###########################
    zero = torch.tensor(0.0).cuda()
    target_expand = target_idx.view(batch_size,1,1,-1).expand(-1,-1,output_len,-1)
    out = torch.where(target_expand==pad, zero, out)
############################
    for cnt, ngram in enumerate(ngram_list):
        if ngram > output_len:
            continue
        eye_filter = torch.eye(ngram).view([1, 1, ngram, ngram]).cuda()
        term = nn.functional.conv2d(out, eye_filter)/ngram
        if ngram < decoder_outputs.size()[1]:
            term = term.squeeze(1)
            # gum_tmp = F.gumbel_softmax(term.squeeze_(1), tau=1, dim=1)
            gum_tmp = F.gumbel_softmax(term, tau=1, dim=1)
            term = term.mul(gum_tmp).sum(1).mean(1)
        else:
            while len(term.shape) > 1:
                assert term.shape[-1] == 1, str(term.shape)
                term = term.sum(-1)
        #print(weight_list[cnt].shape)
        #print(term.shape)
        try:
            sum_gram += weight_list[cnt] * term #.sum()  #TODO
        except:
            print(sum_gram.shape)
            print(term.shape)
            print((weight_list[cnt] * term).shape)
            print(ngram)
            print(decoder_outputs.size()[1])
            assert False

    loss = - sum_gram #/ batch_size

    #return loss.sum()
    return loss

