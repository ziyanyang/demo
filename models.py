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

class CNN_Encoder(nn.Module):
    """
    CNN_Encoder.
    Resnet50.
    """

    def __init__(self, encoded_image_size=14):
        super(CNN_Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = torchvision.models.resnet50(pretrained=True)
        # Remove last linear layer and pooling layers
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images [batch_size, encoded_image_size=14, encoded_image_size=14, 2048]
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # [batch_size, 2048/512, 8, 8] -> [batch_size, 2048/512, 14, 14]
        out = out.permute(0, 2, 3, 1)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for backbone CNN.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False

        for c in list(self.resnet.children()):
            for p in c.parameters():
                p.requires_grad = fine_tune

class LF(nn.Module):
    def __init__(self, encoded_image_size=14):
    
        super(LF, self).__init__()

        self.enc_image_size = encoded_image_size
        resnet = torchvision.models.resnet50(pretrained = True)

        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size)) 
        self.layers = [
            ('input_image', lambda x:x),
            ('conv1', lambda x: self.resnet[0](x)),
            ('conv5', lambda x: self.resnet[4](self.resnet[3](
                self.resnet[2](self.resnet[1](x))))),
            ('conv9',  lambda x: self.resnet[5](x)),
            ('conv13', lambda x: self.resnet[6](x)),
            ('conv17', lambda x: self.resnet[7](x)),
            ('adp_pool', lambda x: self.adaptive_pool(x)),
        ]
            
    def forward(self, x):
        for name, operator in self.layers:
            x = operator(x)
            setattr(self, name, x)
        # Take the max for each prediction map.
        return x.permute(0, 2, 3, 1)
    
    def partial_forward(self, start):
        skip = True
        for name, operator in self.layers:
            if name == start:
                x = getattr(self, name)
                skip = False
            elif skip:
                continue
            else:
                x = operator(x)
                setattr(self, name, x)

        return x.permute(0, 2, 3, 1)

    
class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # [batch_size_t, num_pixels=196, 2048] -> [batch_size_t, num_pixels, attention_dim]
        att2 = self.decoder_att(decoder_hidden)  # [batch_size_t, decoder_dim=512] -> [batch_size_t, attention_dim]
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # [batch_size_t, num_pixels=196, attention_dim] -> [batch_size_t, num_pixels]
        alpha = self.softmax(att)  # [batch_size_t, num_pixels=196]
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # [batch_size_t, encoder_dim=2048]

        return attention_weighted_encoding, alpha

class DecoderWithTripleAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocabs, encoder_dim=2048, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithTripleAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size1 = len(vocabs['EN'])
        self.vocab_size2 = len(vocabs['DE'])
        self.vocab_size3 = len(vocabs['JP'])
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network
        self.embedding = nn.Embedding(self.vocab_size1, embed_dim, padding_idx=0)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, self.vocab_size1)  # linear layer to find scores over vocabulary
        
        self.embedding_sec = nn.Embedding(self.vocab_size2, embed_dim, padding_idx=0)  # embedding layer
        self.decode_step_sec = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h_sec = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c_sec = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta_sec = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.fc_sec = nn.Linear(decoder_dim, self.vocab_size2)  # linear layer to find scores over vocabulary
        
        self.embedding_third = nn.Embedding(self.vocab_size3, embed_dim, padding_idx=0)  # embedding layer
        self.decode_step_third = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h_third = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c_third = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta_third = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.fc_third = nn.Linear(decoder_dim, self.vocab_size3)  # linear layer to find scores over vocabulary

        
        self.init_weights()  # initialize some layers with the uniform distribution
        self.fine_tune_embeddings()

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)
        
        self.embedding_sec.weight.data.uniform_(-0.1, 0.1)
        self.fc_sec.bias.data.fill_(0)
        self.fc_sec.weight.data.uniform_(-0.1, 0.1)
        
        self.embedding_third.weight.data.uniform_(-0.1, 0.1)
        self.fc_third.bias.data.fill_(0)
        self.fc_third.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)
        self.embedding_sec.weight = nn.Parameter(embeddings_sec)
        self.embedding_third.weight = nn.Parameter(embeddings_third)
    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune
        for p in self.embedding_sec.parameters():
            p.requires_grad = fine_tune
        for p in self.embedding_third.parameters():
            p.requires_grad = fine_tune
    def init_hidden_state(self, encoder_out, language):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)  # [batch_size, 196, 2048] -> [batch_size, 2048]
        if language == 'EN':
            h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
            c = self.init_c(mean_encoder_out)
        elif language == 'DE':
            h = self.init_h_sec(mean_encoder_out)  # (batch_size, decoder_dim)
            c = self.init_c_sec(mean_encoder_out)
        elif language == 'JP':
            h = self.init_h_third(mean_encoder_out)  # (batch_size, decoder_dim)
            c = self.init_c_third(mean_encoder_out)
        else:
            assert(0)
        return h, c


    def forward(self, encoder_out, encoded_captions, caption_lengths, language_type):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """
        # [batch_size, 14, 14, 2048]/[batch_size, 196, 2048] -> [batch_size, 196, 2048]
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size1 = self.vocab_size1
        vocab_size2 = self.vocab_size2
        vocab_size3 = self.vocab_size3
        
        # Flatten image -> [batch_size, num_pixels=196, encoder_dim=2048]
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? For each of data in the batch, when len(prediction) = len(caption_lengths), Stop.
        
        if len(caption_lengths.size()) == 1:
            caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        else:
            caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
            
        encoder_out = encoder_out[sort_ind] # sort encoded image based on length of caption 1 languegs

        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        if language_type == 'EN':
            embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)
        elif language_type == 'DE':
            embeddings = self.embedding_sec(encoded_captions)
        elif language_type == 'JP':
            embeddings = self.embedding_third(encoded_captions)
        else:
            print("ERROR! sec_lan is not correct. ")
            

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out, language_type)  # [batch_size, decoder_dim]

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        if language_type == 'EN':  
            predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size1).to(device)
            alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

            for t in range(max(decode_lengths)):
                batch_size_t = sum([l > t for l in decode_lengths])
                attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                    h[:batch_size_t])
                gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
                attention_weighted_encoding = gate * attention_weighted_encoding
                h, c = self.decode_step(
                    torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                    (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
                preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
                predictions[:batch_size_t, t, :] = preds
                alphas[:batch_size_t, t, :] = alpha

        elif language_type == 'DE':   
            predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size2).to(device)
            alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)
            for t in range(max(decode_lengths)):
                batch_size_t = sum([l > t for l in decode_lengths])
                attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                    h[:batch_size_t])
                gate = self.sigmoid(self.f_beta_sec(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
                attention_weighted_encoding = gate * attention_weighted_encoding
                h, c = self.decode_step_sec(
                    torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                    (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
                preds = self.fc_sec(self.dropout(h))  # (batch_size_t, vocab_size)
                predictions[:batch_size_t, t, :] = preds
                alphas[:batch_size_t, t, :] = alpha
                
        elif language_type == 'JP':
            predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size3).to(device)
            alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)
            for t in range(max(decode_lengths)):
                batch_size_t = sum([l > t for l in decode_lengths])
                attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                    h[:batch_size_t])
                gate = self.sigmoid(self.f_beta_third(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
                attention_weighted_encoding = gate * attention_weighted_encoding
                h, c = self.decode_step_third(
                    torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                    (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
                preds = self.fc_third(self.dropout(h))  # (batch_size_t, vocab_size)
                predictions[:batch_size_t, t, :] = preds
                alphas[:batch_size_t, t, :] = alpha    
        else:
            assert(0)

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
    
    def decode(self, k_prev_words, encoder_out, k, target, word_map):
        vocab_size = len(word_map)
        
        seqs = k_prev_words
        top_k_scores = torch.zeros(k, 1).to(device)
        
        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        step = 1
        max_length = False
        
        h, c = self.init_hidden_state(encoder_out, target)
        
        hypotheses = []
#         print('decode...')
        while True:
                

            if target == 'EN':

                embeddings = self.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
                awe, _ = self.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)
                gate = self.sigmoid(self.f_beta(h))  # gating scalar, (s, encoder_dim)
                awe = gate * awe
                h, c = self.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)
                scores = self.fc(h)  # (s, vocab_size)
                scores = F.log_softmax(scores, dim=1)
            elif target == 'DE':
                embeddings = self.embedding_sec(k_prev_words).squeeze(1)  # (s, embed_dim)
                awe, _ = self.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)
                gate = self.sigmoid(self.f_beta_sec(h))  # gating scalar, (s, encoder_dim)
                awe = gate * awe
                h, c = self.decode_step_sec(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)
                scores = self.fc_sec(h)  # (s, vocab_size)
                scores = F.log_softmax(scores, dim=1)
            elif target == 'JP':
                embeddings = self.embedding_third(k_prev_words).squeeze(1)  # (s, embed_dim)
                awe, _ = self.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)
                gate = self.sigmoid(self.f_beta_third(h))  # gating scalar, (s, encoder_dim)
                awe = gate * awe
                h, c = self.decode_step_third(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)
                scores = self.fc_third(h)  # (s, vocab_size)
                scores = F.log_softmax(scores, dim=1)
            else:
                print('Wrong target!')
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)
            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words // vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)
            
            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]

            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]

            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                max_length=True
                break
            step += 1
            
#         if max_length:
#             seq = seqs[0][:20].cpu().tolist()

        if len(complete_seqs_scores) == 0:
            seq = seqs[0][:20].cpu().tolist()
        else:    
            i = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[i]

        hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])    
        return hypotheses
