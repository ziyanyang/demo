import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import json
import torchvision.models as models
# from nltk.translate.bleu_score import corpus_bleu
import pickle

import time
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.optim as optim


import torch
from torch import nn
import torchvision
from collections import OrderedDict

import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
def evaluate_lstm(beam_size, word_map, encoder_outs, target, decoder):
    beam_size = beam_size

    with torch.no_grad():
        enc_image_size = encoder_outs.size(1)
        encoder_dim = encoder_outs.size(-1)
        for id_ in range(0,encoder_outs.size(0)):
            k = beam_size
            encoder_out = encoder_outs[id_].unsqueeze(0).view(1, -1, encoder_dim)
            num_pixels = enc_image_size*enc_image_size
            encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)
            
            k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)
#             print('k_prev_words is ', k_prev_words)
            hypotheses = decoder.decode(k_prev_words, encoder_out, k, target, word_map)
            
        return hypotheses
    
def feedback_test(args, encoder, decoder, img_tensor, sample_caps, sample_caplen, all_word_maps):
    criterion = nn.CrossEntropyLoss().to(device)
    
    img_tensor = img_tensor.to(device) # load and preprocess
    sample_caps = sample_caps.unsqueeze(0).to(device)
    sample_caplen = sample_caplen.unsqueeze(0).to(device)
    imgs = encoder(img_tensor)

    scores1, caps_sorted_l1, decode_lengths1, alphas1, sort_ind1 = decoder(imgs, sample_caps, sample_caplen, args.src)
            
        
    activation = getattr(encoder, args.layer).data

    activation.requires_grad = True

    setattr(encoder, args.layer, activation)

    optimizer = optim.Adam([activation], lr = args.lr, weight_decay = 1e-4)
    for iteration in range(0, args.iteration+1):
        output = encoder.partial_forward(args.layer)
        if iteration == 0:
            init_output = output
        scores1, caps_sorted_l1, decode_lengths1, alphas1, sort_ind1 = decoder(output, sample_caps, sample_caplen, args.src)
        targets_l1 = caps_sorted_l1[:, 1:]
        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores1 = pack_padded_sequence(scores1, decode_lengths1, batch_first=True).data
        targets1 = pack_padded_sequence(targets_l1, decode_lengths1, batch_first=True).data


        if args.decoder_mode=="lstm":
            loss = criterion(scores1, targets1) 
            loss += 1.0 * ((1. - alphas1.sum(dim=1)) ** 2).mean()

        else:   
            loss = criterion(scores1, targets1) 
            dec_alphas = alphas1["dec_enc_attns"]
            alpha_trans_c = 1.0 / (8 * 4)
            for layer in range(4):  # args.decoder_layers = len(dec_alphas)
                cur_layer_alphas = dec_alphas[layer]  # [batch_size, n_heads, 52, 196]
                for h in range(8):
                    cur_head_alpha = cur_layer_alphas[:, h, :, :]
                    loss += alpha_trans_c * ((1. - cur_head_alpha.sum(dim=1)) ** 2).mean()
#         print(loss)
        optimizer.zero_grad()  
        loss.backward()
        optimizer.step()
            
    activation.requires_grad = False 
    output = encoder.partial_forward(args.layer)

#     print(init_output.shape)
#     print(output.shape)
    if args.decoder_mode == 'lstm':
        hypothesis_ori = evaluate_lstm(3, all_word_maps[args.tgt], init_output, args.tgt, decoder)
        hypothesis = evaluate_lstm(3, all_word_maps[args.tgt], output, args.tgt, decoder)
#     else:
#         hypothesis = evaluate_transformer(3, all_word_maps[args.tgt], output, args.tgt, decoder)

    return hypothesis_ori, hypothesis 


