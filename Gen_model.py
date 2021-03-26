import logging
logger = logging.getLogger()
logger.setLevel("ERROR")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import math
import json
import copy
import random
from tqdm import tqdm
from transformers import *


f = open('./merged_vocab.json')
token2num = json.load(f)

num2token = {}
for key, value in token2num.items():
    num2token[value] = key

vocab_size = len(token2num.keys())


##### Hanlp Tokenizer #####
import hanlp
HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH) # 世界最大中文语料库

def Hanlp_tokenizer(text_str:str, vocab_dict=token2num)->list:
    seg_dict = HanLP(text_str, tasks=['tok/fine'])
    seg_tokens = seg_dict['tok/fine']
    seg_ids = [token2num.get(seg, 0) for seg in seg_tokens]
    return seg_ids


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class styletransfer(nn.Module):
    def __init__(self, vocab_size, tokenizer=Hanlp_tokenizer, drop_rate=0):
        super(styletransfer, self).__init__()
        self.DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = tokenizer
        # self.gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        """hyper parameters"""
        self.n_vocab = vocab_size
        self.emb_dim = 128
        self.nhead = 4
        self.num_layers = 3

        """idx & length"""
        self.START_IDX = 1
        self.PAD_IDX = 0
        self.EOS_IDX = 2
        self.MAX_SENT_LEN = 10

        """attribute matrix"""
        ## one_hot encoding
        self.att_num = 2  # Attribute categories
        self.matrix_A = nn.Linear(self.att_num, self.emb_dim)

        """word embedding"""
        self.emb_matrix = nn.Embedding(self.n_vocab, self.emb_dim, self.PAD_IDX)

        """Position embedding"""
        self.pos_encoder = PositionalEncoding(self.emb_dim)

        """Encoder"""
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.emb_dim, nhead=self.nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)

        """Decoder"""
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.emb_dim, nhead=self.nhead)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=self.num_layers)
        self.matrix_D = nn.Linear(self.emb_dim, self.n_vocab)  # emb_dim -> n_vocab

        """parameters"""
        self.enc_params = list(self.encoder_layer.parameters()) + list(self.transformer_encoder.parameters())
        self.dec_params = list(self.decoder_layer.parameters()) + list(self.transformer_decoder.parameters()) + list(
            self.matrix_D.parameters())
        self.aed_params = list(self.emb_matrix.parameters()) + self.enc_params + self.dec_params

    """Modeling"""

    def encoder(self, enc_input):
        """
        enc_input: (batch, enc_len)
        """
        word_emb = self.emb_matrix(enc_input)  # (batch, enc_len, emb_dim)
        word_emb = word_emb.transpose(0, 1)  # (enc_len, batch, emb_dim)
        word_pos = self.pos_encoder(word_emb)  # (enc_len, batch, emb_dim)
        out_enc = self.transformer_encoder(word_pos)  # (enc_len, batch, emb_dim)

        return out_enc

    def decoder(self, enc_out, dec_input, attribute):
        """
        enc_out: (enc_len, batch, emb_dim)
        dec_input: (batch, dec_len)
        attributes: (batch, 2)
        """
        att_emb = self.matrix_A(attribute).unsqueeze(0)  # (1. batch, emb_dim)

        word_emb = self.emb_matrix(dec_input)  # (batch, dec_len, emb_dim)
        word_emb = word_emb.transpose(0, 1)  # (dec_len, batch, emb_dim)
        word_pos = self.pos_encoder(word_emb)  # (dec_len, batch, emb_dim)

        start_token = self.emb_matrix(torch.tensor(self.START_IDX).to(self.DEVICE))  # (emb_dim)
        start_token = start_token.repeat(1, dec_input.shape[0], 1)  # (1, batch, emb_dim)
        style_dec_input = torch.cat([att_emb, start_token, word_pos],
                                    0)  # (dec_len+2, batch, emb_dim) w/ [att], [start]

        tgt_mask = self.generate_square_subsequent_mask(style_dec_input.shape[0]).to(
            self.DEVICE)  # (dec_len+2, dec_len+2)

        dec_out = self.transformer_decoder(style_dec_input, enc_out, tgt_mask=tgt_mask)  # (dec_len+2, batch, emb_dim)
        vocab_out = self.matrix_D(dec_out)  # (dec_len+2, batch, n_vocab)
        return dec_out, vocab_out

    def generator(self, enc_out, gen_len, attribute):
        """
        enc_out: (enc_len, batch, emb_dim)
        attributes: (batch, 2)
        gen_len: len(dec_in)+1
        """
        # initialization because there are no first token
        batch = enc_out.shape[1]
        att_emb = self.matrix_A(attribute).unsqueeze(0)  # (1. batch, emb_dim)
        start_token = self.emb_matrix(torch.tensor(self.START_IDX).to(self.DEVICE))  # (emb_dim)
        start_token = start_token.repeat(1, batch, 1)  # (1, batch, emb_dim)
        gen_input = torch.cat([att_emb, start_token], 0)  # (2, batch, emb_dim) w/ [att], [start]

        for i in range(gen_len):
            tgt_mask = self.generate_square_subsequent_mask(gen_input.shape[0]).to(
                self.DEVICE)  # (pre_gen_len, pre_gen_len)
            dec_out = self.transformer_decoder(gen_input, enc_out, tgt_mask=tgt_mask)  # (pre_gen_len, batch, emb_dim)
            vocab_out = self.matrix_D(dec_out)  # (pre_gen_len, batch, n_vocab)

            vocab_idx = vocab_out.argmax(2)  # (pre_gen_len, batch)
            vocab_idx = vocab_idx.transpose(0, 1)  # (batch, pre_gen_len)

            new_word_emb = self.emb_matrix(vocab_idx)  # (batch, pre_gen_len, emb_dim)
            new_word_emb = new_word_emb.transpose(0, 1)  # (pre_gen_len, batch, emb_dim)
            #             gen_emb = torch.bmm(vocab_out, self.emb_matrix.weight.repeat(vocab_out.shape[0],1,1))

            #             word_pos = self.pos_encoder(word_emb) # (enc_len, batch, emb_dim)
            gen_input = torch.cat(
                [gen_input, new_word_emb[-1:, :, :]])  # (pre_gen_len+1, batch, word_dim), pre_gen_len+=1

        return vocab_out  # (gen_len+2, batch, n_vocab)

    def generate_square_subsequent_mask(self, sz):  # len(sz)
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    """calculation loss"""

    def recon_loss(self, dec_input, vocab_out):
        """
        dec_input: (batch, dec_len)
        vocab_out: (dec_len+2, batch, n_vocab) with [att], [start]
        """
        end_token = torch.tensor(self.EOS_IDX).to(self.DEVICE)  # (1)
        end_token = end_token.repeat(dec_input.shape[0], 1)  # (batch, 1)
        target_tokens = torch.cat([dec_input, end_token], 1)  # (batch, dec_len+1) w/ [EOS]

        pred_out = vocab_out[1:, :, :]  # (dec_len+1, batch, n_vocab)
        pred_out = pred_out.permute(1, 0, 2)  # (batch, dec_len+1, n_vocab)

        target_tokens = target_tokens.contiguous()  # (batch, dec_len+1)
        pred_out = pred_out.contiguous()  # (batch, dec_len+1, n_vocab)

        target_tokens = target_tokens.view(-1)  # (batch*(dec_len+1))
        pred_out = pred_out.view(-1, pred_out.shape[2])  # (batch*(seq_len+1), n_vocab)

        recon_loss = F.cross_entropy(pred_out, target_tokens)

        return recon_loss

    def cls_loss(self, attributes, cls_out):
        """
        attributes: [0,1] or [1,0]
        cls_out: (batch, 2) (logits)
        """
        targets = attributes.argmax(1)  # (batch)
        cls_loss = F.cross_entropy(cls_out, targets)

        return cls_loss.to(self.DEVICE)

    """inferenece"""

    def dec2sen(self, vocab_out):
        """
        vocab_out: (dec_len+2, batch, n_vocab) with att, start
        """
        pred_out = vocab_out[1:, :, :]  # (dec_len+1, batch, n_vocab) with [END]
        pred_idx = torch.argmax(pred_out, 2)  # (dec_len+1, batch)
        pred_idx = pred_idx.squeeze(1)  # (dec_len+1) because of batch=1

        token_list = []
        dec_sen = ''
        for i in range(len(pred_idx)):
            token = num2token[pred_idx[i].cpu().numpy().item()]
            token_list.append(token)
            dec_sen += token
        dec_sen = dec_sen.strip()
        return token_list, dec_sen

    def generated_sentence(self, enc_out, attribute, ori_length):
        """
        enc_out: (enc_len, batch, emb_dim)
        dec_input: (batch, dec_len)
        attributes: (batch, 2)
        """
        batch = enc_out.shape[1]
        #         max_len = enc_out.shape[0]+3
        max_len = ori_length + 5

        # initialization because there are no first token
        att_emb = self.matrix_A(attribute).unsqueeze(0)  # (1. batch, emb_dim)
        start_token = self.emb_matrix(torch.tensor(self.START_IDX).to(self.DEVICE))  # (emb_dim)
        start_token = start_token.repeat(1, batch, 1)  # (1, batch, emb_dim)
        gen_input = torch.cat([att_emb, start_token], 0)  # (2, batch, emb_dim) w/ [att], [start]

        tgt_mask = self.generate_square_subsequent_mask(gen_input.shape[0]).to(self.DEVICE)  # (2, 2)

        dec_out = self.transformer_decoder(gen_input, enc_out, tgt_mask=tgt_mask)  # (2, batch, emb_dim)
        vocab_out = self.matrix_D(dec_out)  # (2, batch, n_vocab)
        _, dec_sen = self.dec2sen(vocab_out)

        gen_vocab_out = []
        for i in range(max_len):
            if len(dec_sen) == 0:
                token_idx = torch.tensor([220]).unsqueeze(0).to(self.DEVICE)  # (batch, gen_len)
            else:
                token_idx = torch.tensor(self.tokenizer(dec_sen)).unsqueeze(0).to(
                    self.DEVICE)  # (batch, gen_len)
            if self.EOS_IDX in token_idx:
                break

            dec_out, vocab_out = self.decoder(enc_out, token_idx,
                                              attribute)  # (dec_len+2, batch, emb_dim), (dec_len+2, batch, n_vocab)
            dec_tokens, dec_sen = self.dec2sen(vocab_out)

        return dec_sen


class findattribute(nn.Module):
    def __init__(self, vocab_size, drop_rate=0):
        super(findattribute, self).__init__()
        self.DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.n_vocab = vocab_size
        self.emb_dim = 128
        self.drop_rate = drop_rate

        """idx & length"""
        self.EOS_IDX = 2
        self.START_IDX = 1
        self.PAD_IDX = 0

        """Discriminator(classifier)"""
        self.word_dim = 128
        self.word_emb = nn.Embedding(self.n_vocab, self.word_dim, self.PAD_IDX)  # vocab_size x 256

        self.channel_out = 40
        self.conv2d_2 = nn.Conv2d(1, self.channel_out, kernel_size=(2, self.word_dim))
        self.conv2d_3 = nn.Conv2d(1, self.channel_out, kernel_size=(3, self.word_dim))
        self.conv2d_4 = nn.Conv2d(1, self.channel_out, kernel_size=(4, self.word_dim))
        self.conv2d_5 = nn.Conv2d(1, self.channel_out, kernel_size=(5, self.word_dim))
        self.fc_drop = nn.Dropout(self.drop_rate)
        self.disc_fc = nn.Linear(4 * self.channel_out, 2)

        """parameters"""
        self.cls_params = list(self.word_emb.parameters()) + list(self.conv2d_2.parameters()) + list(
            self.conv2d_3.parameters()) + list(self.conv2d_4.parameters()) + \
                          list(self.conv2d_5.parameters()) + list(self.disc_fc.parameters())

    def discriminator(self, token_idx):
        """
        Applying `Sliding Window`?
        token_idx: (batch, seq_len)
        """
        if token_idx.shape[1] < 5:
            padding_size = 5 - token_idx.shape[1]
            padding_token = []
            for k in range(token_idx.shape[0]):
                temp = []
                for i in range(padding_size):
                    temp.append(self.PAD_IDX)
                padding_token.append(temp)
            padding_token = torch.from_numpy(np.array(padding_token))

            padding_token = padding_token.to(self.DEVICE)
            token_idx = torch.cat([token_idx, padding_token], dim=1)  # (batch, seq_len+padding) = (batch, 5)

        word_emb = self.word_emb(token_idx)  # (batch, seq_len, word_dim)
        word_2d = word_emb.unsqueeze(1)  # (batch, 1, seq_len, word_dim)

        x2 = F.relu(self.conv2d_2(word_2d)).squeeze(3)  # bi-gram, (batch, channel_out, seq_len-1)
        x3 = F.relu(self.conv2d_3(word_2d)).squeeze(3)  # 3-gram, (batch, channel_out, seq_len-2)
        x4 = F.relu(self.conv2d_4(word_2d)).squeeze(3)  # 4-gram, (batch, channel_out, seq_len-3)
        x5 = F.relu(self.conv2d_5(word_2d)).squeeze(3)  # 5-gram, (batch, channel_out, seq_len-4)

        # Max-over-time-pool
        x2 = F.max_pool1d(x2, x2.size(2)).squeeze(2)  # (batch, channel_out)
        x3 = F.max_pool1d(x3, x3.size(2)).squeeze(2)  # (batch, channel_out)
        x4 = F.max_pool1d(x4, x4.size(2)).squeeze(2)  # (batch, channel_out)
        x5 = F.max_pool1d(x5, x5.size(2)).squeeze(2)  # (batch, channel_out)
        x = torch.cat([x2, x3, x4, x5], dim=1)  # (batch, channel_out*4)

        x_drop = self.fc_drop(x)
        y = self.disc_fc(x_drop)  # (batch, 2)

        return y.to(self.DEVICE)

    ## inference
    def gen_discriminator(self, gen_out):
        """
        gen_out: (gen_len+2, batch, n_vocab)
        """
        gen_emb = gen_out[1:-1, :, :]  # (gen_len, batch, n_vocab)
        gen_emb = torch.bmm(gen_emb, self.word_emb.weight.repeat(gen_emb.shape[0], 1, 1))
        # (gen_len, batch, emb_dim) = (gen_len, batch, n_vocab) x (gen_len, n_vocab, emb_dim)
        gen_emb = gen_emb.transpose(0, 1)  # (batch, gen_len, word_dim)

        if gen_emb.shape[1] < 5:
            padding_size = 5 - gen_emb.shape[1]
            padding_token = []
            for k in range(gen_emb.shape[0]):
                temp = []
                for i in range(padding_size):
                    temp.append(self.PAD_IDX)
                padding_token.append(temp)
            padding_token = torch.from_numpy(np.array(padding_token))  # (batch, padding_len)

            padding_token = padding_token.to(self.DEVICE)
            padding_emb = self.word_emb(padding_token)  # (batch, padding_len, emb_dim)
            gen_emb = torch.cat([gen_emb, padding_emb], 1)  # (batch, 5, emb_dim)

        word_2d = gen_emb.unsqueeze(1)  # (batch, 1, seq_len, word_dim)

        x2 = F.relu(self.conv2d_2(word_2d)).squeeze(3)  # bi-gram, (batch, channel_out, seq_len-1)
        x3 = F.relu(self.conv2d_3(word_2d)).squeeze(3)  # 3-gram, (batch, channel_out, seq_len-2)
        x4 = F.relu(self.conv2d_4(word_2d)).squeeze(3)  # 4-gram, (batch, channel_out, seq_len-3)
        x5 = F.relu(self.conv2d_5(word_2d)).squeeze(3)  # 5-gram, (batch, channel_out, seq_len-4)

        # Max-over-time-pool
        x2 = F.max_pool1d(x2, x2.size(2)).squeeze(2)  # (batch, channel_out)
        x3 = F.max_pool1d(x3, x3.size(2)).squeeze(2)  # (batch, channel_out)
        x4 = F.max_pool1d(x4, x4.size(2)).squeeze(2)  # (batch, channel_out)
        x5 = F.max_pool1d(x5, x5.size(2)).squeeze(2)  # (batch, channel_out)
        x = torch.cat([x2, x3, x4, x5], dim=1)  # (batch, channel_out*4)

        y = self.disc_fc(x)  # (batch, 2)

        return y.to(self.DEVICE)

    def att_prob(self, token_idx, sentiment):
        """
        token_idx: (batch, seq_len)
        """
        token_list = token_idx.squeeze(0).cpu().tolist()  # list
        min_prob = 1
        for i in range(len(token_list)):
            del_list = token_list[:i] + token_list[i + 1:]
            del_tensor = torch.from_numpy(np.asarray(del_list)).unsqueeze(0).to(self.DEVICE)
            del_prob = F.softmax(self.discriminator(del_tensor), 1).squeeze(0)[sentiment].cpu().detach().numpy().item()

            if del_prob <= min_prob:
                max_ind = i
                min_prob = del_prob

        final_list = token_list[:max_ind] + token_list[max_ind + 1:]
        del_idx = torch.from_numpy(np.asarray(final_list)).unsqueeze(0).to(self.DEVICE)
        return del_idx

    def cls_loss(self, targets, cls_out):
        """
        targets: (batch, 2) / attributes [0,1] or [1,0]
        cls_out: (batch, 2) (logits)
        """
        final_targets = targets.argmax(1)  # (batch)
        cls_loss = F.cross_entropy(cls_out, final_targets)

        return cls_loss.to(self.DEVICE)


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

genmodel = styletransfer(vocab_size=vocab_size).to(DEVICE)
# genmodel.load_state_dict(torch.load('../ST_v2.0/models/gen_model_5'))
genmodel.train()

import sys

# sys.path.insert(0, "/DATA/joosung/controllable_english/amazon/classifier/")
dismodel = findattribute(vocab_size=vocab_size).to(DEVICE)
dismodel_name = 'cls_model_5'
dismodel.load_state_dict(torch.load('./models/{}'.format(dismodel_name)))
dismodel.eval()

from tensorboardX import SummaryWriter

summary = SummaryWriter(logdir='./logs')

##### Load Data #####
news_train_ids_json = './news_train_ids.json'
news_val_ids_json = './news_val_ids.json'
poems_train_ids_json = './poems_train_ids.json'
poems_val_ids_json = './poems_val_ids.json'

with open(news_train_ids_json, 'r', encoding='utf-8') as news_trg:
    news_train_ids = json.load(news_trg)

with open(news_val_ids_json, 'r', encoding='utf-8') as news_trg:
    news_val_ids = json.load(news_trg)

with open(poems_train_ids_json, 'r', encoding='utf-8') as poems_trg:
    poems_train_ids = json.load(poems_trg)

with open(poems_val_ids_json, 'r', encoding='utf-8') as poems_trg:
    poems_val_ids = json.load(poems_trg)

print('Data Loaded!')

recon_loss_list, gen_cls_loss_list = [], []


def main():
    merge_dict_json = './merged_vocab.json'
    with open(merge_dict_json, 'r', encoding='utf-8') as dict_json:
        token2num = json.load(dict_json)

    num2token = {}
    for key, value in token2num.items():
        num2token[value] = key
    print('Corpus Loaded!')

    train_news_set, val_news_set, train_poems_set, val_poems_set = [news for news in news_train_ids if
                                                                    len(news) < 50], [news for news in news_val_ids if
                                                                                      len(news) < 50], [poems for poems
                                                                                                        in
                                                                                                        poems_train_ids
                                                                                                        if len(
            poems) < 50], [poems for poems in poems_val_ids if len(poems) < 50]
    total_news_set = train_news_set + val_news_set
    news_set = random.sample(total_news_set, len(total_news_set) // 20)
    poems_set = train_poems_set + val_poems_set
    news_len, poems_len = len(news_set), len(poems_set)
    print('#News Lines:', news_len)
    print('#Poem Lines:', poems_len)

    """training parameter"""
    aed_initial_lr = 0.00001
    gen_initial_lr = 0.001
    ### aed_trainer: Auto-Encoder--> Find features
    aed_trainer = optim.Adamax(genmodel.aed_params, lr=aed_initial_lr)  # initial 0.0005
    gen_trainer = optim.Adamax(genmodel.aed_params, lr=gen_initial_lr)  # initial 0.0001
    max_grad_norm = 10
    BATCH_SIZE = 1
    NUM_EPOCH = 6
    epoch_len = min(poems_len, news_len)
    stop_point = epoch_len * NUM_EPOCH

    pre_epoch = 0
    running_recon_loss, running_gen_loss = 0.0, 0.0
    for start in tqdm(range(0, stop_point)):
        ## learing rate decay
        now_epoch = (start + 1) // news_len

        """data start point"""
        news_start = start % news_len
        poems_start = start % poems_len

        """data setting"""
        news_seq = news_set[news_start]
        poems_seq = poems_set[poems_start]

        news_labels = []  # news labels
        news_labels.append([1, 0])
        news_attribute = torch.from_numpy(np.asarray(news_labels)).type(torch.FloatTensor).to(DEVICE)

        poems_labels = []  # poems labels
        poems_labels.append([0, 1])
        poems_attribute = torch.from_numpy(np.asarray(poems_labels)).type(torch.FloatTensor).to(DEVICE)

        seqs = [news_seq, poems_seq]
        attributes = [news_attribute, poems_attribute]
        styles = [0, 1]

        """data input"""
        for i in range(2):
            # k=0: negative, k=1: positive
            seq = seqs[i]
            attribute = attributes[i]  # for decoder
            fake_attribute = attributes[abs(1 - i)]  # for generate
            #             sentiment = sentiments[i] # for delete

            token_idx = torch.tensor(seq).unsqueeze(0).to(DEVICE)

            # delete model
            max_len = int(token_idx.shape[1] / 2)
            dis_out = dismodel.discriminator(token_idx)
            style = dis_out.argmax(1).cpu().item()

            del_idx = token_idx
            for k in range(max_len):
                del_idx = dismodel.att_prob(del_idx, style)
                dis_out = dismodel.discriminator(del_idx)
                sent_porb = F.softmax(dis_out, 1).squeeze(0)[style].cpu().detach().numpy().item()
                if sent_porb < 0.7:
                    break

            """auto-encoder loss & traning"""
            # training using discriminator loss
            enc_out = genmodel.encoder(del_idx)
            dec_out, vocab_out = genmodel.decoder(enc_out, token_idx, attribute)

            ## calculation loss
            recon_loss = genmodel.recon_loss(token_idx, vocab_out)
            summary.add_scalar('reconstruction loss', recon_loss.item(), start)
            aed_trainer.zero_grad()

            ## calculation loss
            gen_cls_out = dismodel.gen_discriminator(vocab_out)
            gen_cls_loss = genmodel.cls_loss(attribute, gen_cls_out)
            summary.add_scalar('generated sentence loss', gen_cls_loss.item(), start)
            gen_trainer.zero_grad()

            running_recon_loss += recon_loss.detach().item()
            running_gen_loss += gen_cls_loss.detach().item()

            recon_loss.backward(retain_graph=True)  # retain_graph=True
            gen_cls_loss.backward()  # retain_graph=True
            grad_norm = torch.nn.utils.clip_grad_norm_(genmodel.aed_params, max_grad_norm)

            aed_trainer.step()
            gen_trainer.step()

        """savining point"""
        if (start + 1) % epoch_len == 0:
            recon_loss_list.append(running_recon_loss / (len(total_news_set) // 20))
            gen_cls_loss_list.append(running_gen_loss / (len(total_news_set) // 20))
            running_recon_loss, running_gen_loss = 0.0, 0.0
            print(
                f'Epoch: {(start + 1) // epoch_len} | Recon_loss: {running_recon_loss / (len(total_news_set) // 20)} | Gen_cls_loss:{running_gen_loss / (len(total_news_set) // 20)}')
            news_set = random.sample(total_news_set, len(total_news_set) // 20)
            #             random.shuffle(news_set)
            random.shuffle(poems_set)
            save_model(genmodel, (start + 1) // poems_len)
    save_model(genmodel, 'final')  # final_model


def save_model(gen_model, iter):
    if not os.path.exists('./Generator/models/'):
        os.makedirs('./Generator/models/')
    torch.save(gen_model.state_dict(), './Generator/models/gen_model_{}'.format(iter))


if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()