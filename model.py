import random
import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import DataLoader,RandomSampler,Dataset
from tqdm import tqdm, trange
from transformers import BertModel,BertTokenizer,AdamW, BertConfig, get_linear_schedule_with_warmup,AlbertConfig,AlbertModel
from sklearn.metrics import f1_score,precision_score,recall_score,confusion_matrix
from copy import deepcopy
import torch.nn.functional as F
import pandas as pd
from collections import Counter
from sklearn import metrics
from sklearn.manifold import TSNE

class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate = 0.1):
        super(FCLayer, self).__init__()

        self.dropout1 = nn.Dropout(dropout_rate)
        self.linear1 = nn.Linear(input_dim, output_dim)

        
    def forward(self, x):
        x = self.linear1(self.dropout1(x))
        
        return x
    

class NEEDED2(nn.Module):
    def __init__(self,config, dropout_rate = 0.1):
        super(NEEDED2, self).__init__()
        #self.bert_positive = AlbertModel(config = config)
        #self.bert_negative = BertModel(config = config)
        #self.bert_c = model_c
        
        self.num_labels = config.num_labels
        self.dropout_rate = dropout_rate
        self.embed = nn.Embedding(21128, config.hidden_size, padding_idx = 0)
#         self.cls_fc_layer_1 = FCLayer(config.hidden_size, config.hidden_size,dropout_rate = self.dropout_rate)
#         self.cls_fc_layer_2 = FCLayer(config.hidden_size, config.hidden_size,dropout_rate = self.dropout_rate)
#         self.cls_fc_layer_3 = FCLayer(config.hidden_size, config.hidden_size,dropout_rate = self.dropout_rate)
       
#         self.tanh = nn.Tanh()
#         self.relu = nn.ReLU()
        self.device = config.device
        self.bai_classifier = FCLayer(
            config.hidden_size * 1,
            1,
            self.dropout_rate,
        )
        
        self.qing_classifier = FCLayer(
            config.hidden_size * 1,
            1,
            self.dropout_rate,
        )
        
        self.huang_classifier = FCLayer(
            config.hidden_size * 1,
            1,
            self.dropout_rate,
        )
        self.tang_classifier = FCLayer(
            config.hidden_size * 1,
            1,
            self.dropout_rate,
        )
        self.yi_classifier = FCLayer(
            config.hidden_size * 1,
            1,
            self.dropout_rate,
        )
        self.lei_classifier = FCLayer(
            config.hidden_size * 1,
            1,
            self.dropout_rate,
        )
        self.gan_classifier = FCLayer(
            config.hidden_size * 1,
            1,
            self.dropout_rate,
        )
        
        
        # label embedding 与初始化
        self.bai_label = nn.Parameter(torch.Tensor(config.hidden_size,1))   
        self.qing_label = nn.Parameter(torch.Tensor(config.hidden_size,1))    
        self.huang_label = nn.Parameter(torch.Tensor(config.hidden_size,1))
        self.tang_label = nn.Parameter(torch.Tensor(config.hidden_size,1))
        self.yi_label = nn.Parameter(torch.Tensor(config.hidden_size,1))
        self.lei_label = nn.Parameter(torch.Tensor(config.hidden_size,1))
        self.gan_label = nn.Parameter(torch.Tensor(config.hidden_size,1))
        
        nn.init.uniform_(self.bai_label, -0.1, 0.1)
        nn.init.uniform_(self.qing_label, -0.1, 0.1)
        nn.init.uniform_(self.huang_label, -0.1, 0.1)
        nn.init.uniform_(self.tang_label, -0.1, 0.1)
        nn.init.uniform_(self.yi_label, -0.1, 0.1)
        nn.init.uniform_(self.lei_label, -0.1, 0.1)
        nn.init.uniform_(self.gan_label, -0.1, 0.1)
        
        # 注意softmax的dim
        self.pos_nega_att_softmax = nn.Softmax(dim = 1)
        self.label_att_softmax = nn.Softmax(dim = 0)
        
        self.vec_multi = nn.MultiheadAttention(config.hidden_size, 8, batch_first = True)
        
        # transfomer layer
        trans_layer = torch.nn.TransformerEncoderLayer(config.hidden_size, nhead = 8, batch_first = True, activation = "gelu")
        self.transformer1 = torch.nn.TransformerEncoder(encoder_layer = trans_layer, num_layers = 2)
        self.transformer2 = torch.nn.TransformerEncoder(encoder_layer = trans_layer, num_layers = 1)

    def forward(self, sym_input_ids, sym_attention_mask,sym_token_type_ids):
             
        # all positive encoding
        
        vec_pool_tuple =  self.all_sym_vec(sym_input_ids, sym_token_type_ids)
        all_organ_vec_bai,all_organ_vec_qing,all_organ_vec_tang,all_organ_vec_yi,all_organ_vec_lei,all_organ_vec_gan = vec_pool_tuple 
        
        logits_bai = self.bai_classifier(all_organ_vec_bai)
        logits_qing = self.qing_classifier(all_organ_vec_qing)
        logits_tang = self.tang_classifier(all_organ_vec_tang)
        logits_yi = self.yi_classifier(all_organ_vec_yi)
        logits_lei = self.lei_classifier(all_organ_vec_lei)
        logits_gan = self.gan_classifier(all_organ_vec_gan)
        
        logits = torch.cat([logits_qing,logits_tang,logits_bai,logits_yi,logits_lei,logits_gan],dim = -1)
                
        return logits
    
    # 获得一个样本的 各个sym表示
    def sample_sym_vec(self, one_sym_input_ids, one_token_type_ids, transformer):
        # 求一个样本的positive_sym_num
        one_sym_input_ids = one_sym_input_ids.cpu().numpy().tolist()
        a = one_token_type_ids.cpu().numpy().tolist()
        g = [a[:1]]
        [g[-1].append(y) if x == y else g.append([y]) for x, y in zip(a[:-1], a[1:])]
        
        # the element of one_all_sym_inputs is a list of inputs_id of one sym
        one_all_sym_inputs = []
        i = 0

        for li_ in g:
            # [cls] [sep]
            one_all_sym_inputs.append([101] + one_sym_input_ids[i : i + len(li_)])
            i += len(li_)
        
        one_all_sym_inputs = one_all_sym_inputs[1:]
        one_all_sym_inputs = [x if len(x) <= 64 else x[:64] for x in one_all_sym_inputs]
        #sym_attention_mask = torch.tensor([[1] * len(x) + [0] * (64 - len(x)) for x in one_all_sym_inputs],dtype = torch.float).cuda(0)
        one_all_sym_inputs = torch.tensor([x  + [0] * (64 - len(x)) for x in one_all_sym_inputs],dtype = torch.long).to(self.device)
        
        one_all_sym_inputs = self.embed(one_all_sym_inputs)
        transformer_result = transformer(one_all_sym_inputs)
        vec_pool = torch.mean(transformer_result, dim = 1)
        return vec_pool
    
    # 获得所有样本的postive表示 
    def all_sym_vec(self, sym_input_ids, sym_token_type_ids):
        all_organ_vec_bai = None
        all_organ_vec_qing = None
        all_organ_vec_tang = None
        all_organ_vec_yi = None
        all_organ_vec_lei = None
        all_organ_vec_gan = None
        
        for i in range(sym_input_ids.shape[0]):
            positive_vec_pool = self.sample_sym_vec(sym_input_ids[i], sym_token_type_ids[i], self.transformer1)
            #negative_vec_pool = self.sample_sym_vec(no_sym_input_ids[i], no_sym_token_type_ids[i], self.transformer1)
            
            #pos_nega_vec = self.pos_nega_attention(positive_vec_pool, negative_vec_pool)[0]
            
            #vec_pool = torch.cat((positive_vec_pool, negative_vec_pool),dim = 0)
            vec_pool = positive_vec_pool.unsqueeze(0)
            vec_pool = self.transformer2(vec_pool).squeeze(0)
            #vec_pool = torch.cat([vec_pool, positive_vec_pool],dim = 0)
            
            organ_ave_bai = self.label_wise_attention(vec_pool, self.bai_label)[0]
            organ_ave_qing = self.label_wise_attention(vec_pool, self.qing_label)[0]
            organ_ave_tang = self.label_wise_attention(vec_pool, self.tang_label)[0]
            organ_ave_yi = self.label_wise_attention(vec_pool, self.yi_label)[0]
            organ_ave_lei = self.label_wise_attention(vec_pool, self.lei_label)[0]
            organ_ave_gan = self.label_wise_attention(vec_pool, self.gan_label)[0]
            
            if all_organ_vec_bai == None:
                all_organ_vec_bai = organ_ave_bai
                all_organ_vec_qing = organ_ave_qing
                all_organ_vec_tang = organ_ave_tang
                all_organ_vec_yi = organ_ave_yi
                all_organ_vec_lei = organ_ave_lei
                all_organ_vec_gan = organ_ave_gan
                
            else:
                all_organ_vec_bai = torch.cat((all_organ_vec_bai,organ_ave_bai),dim = 0)
                all_organ_vec_qing = torch.cat((all_organ_vec_qing,organ_ave_qing),dim = 0)
                all_organ_vec_tang = torch.cat((all_organ_vec_tang,organ_ave_tang),dim = 0)
                all_organ_vec_yi = torch.cat((all_organ_vec_yi,organ_ave_yi),dim = 0)
                all_organ_vec_lei = torch.cat((all_organ_vec_lei,organ_ave_lei),dim = 0)
                all_organ_vec_gan = torch.cat((all_organ_vec_gan,organ_ave_gan),dim = 0)
               
        
        return all_organ_vec_bai,all_organ_vec_qing,all_organ_vec_tang,all_organ_vec_yi,all_organ_vec_lei,all_organ_vec_gan        
    


        
    def label_wise_attention(self, vec_pool, label_embedding):
        att_intern = torch.matmul(vec_pool, label_embedding)
        att_score = self.label_att_softmax(att_intern)
        att_out = torch.matmul(att_score.T, vec_pool)
        
        return att_out, att_score
    
    def pos_nega_attention(self, positive_vec_pool, negative_vec_pool):
        att_intern = torch.matmul(positive_vec_pool, negative_vec_pool.T)
        att_score = self.pos_nega_att_softmax(att_intern)
        att_out = torch.matmul(att_score, negative_vec_pool)
        
        return att_out, att_score
    
#     def attention_net_2(self,organ_ave_pool,w_omega,u_omega):
#         u = torch.tanh(torch.matmul(organ_ave_pool, w_omega))
#         attention = torch.matmul(u, u_omega)
#         att_score = F.softmax(attention, dim=0)
#         scored_x = organ_ave_pool * att_score
#         att_out = torch.sum(scored_x, dim = 0) #加权求和

#         return att_out,att_score


class NEEDED5(nn.Module):
    def __init__(self,config, dropout_rate = 0.1):
        super(NEEDED5, self).__init__()
        #self.bert_positive = AlbertModel(config = config)
        #self.bert_negative = BertModel(config = config)
        #self.bert_c = model_c
        
        self.num_labels = config.num_labels
        self.dropout_rate = dropout_rate
        self.embed = nn.Embedding(21128, config.hidden_size, padding_idx = 0)
#         self.cls_fc_layer_1 = FCLayer(config.hidden_size, config.hidden_size,dropout_rate = self.dropout_rate)
#         self.cls_fc_layer_2 = FCLayer(config.hidden_size, config.hidden_size,dropout_rate = self.dropout_rate)
#         self.cls_fc_layer_3 = FCLayer(config.hidden_size, config.hidden_size,dropout_rate = self.dropout_rate)
       
#         self.tanh = nn.Tanh()
#         self.relu = nn.ReLU()
        self.device = config.device
        self.bai_classifier = FCLayer(
            config.hidden_size * 1,
            1,
            self.dropout_rate,
        )
        
        self.qing_classifier = FCLayer(
            config.hidden_size * 1,
            1,
            self.dropout_rate,
        )
        
        self.huang_classifier = FCLayer(
            config.hidden_size * 1,
            1,
            self.dropout_rate,
        )
        self.tang_classifier = FCLayer(
            config.hidden_size * 1,
            1,
            self.dropout_rate,
        )
        self.yi_classifier = FCLayer(
            config.hidden_size * 1,
            1,
            self.dropout_rate,
        )
        self.lei_classifier = FCLayer(
            config.hidden_size * 1,
            1,
            self.dropout_rate,
        )
        self.gan_classifier = FCLayer(
            config.hidden_size * 1,
            1,
            self.dropout_rate,
        )
        
        
        # label embedding 与初始化
        self.bai_label = nn.Parameter(torch.Tensor(config.hidden_size,1))   
        self.qing_label = nn.Parameter(torch.Tensor(config.hidden_size,1))    
        self.huang_label = nn.Parameter(torch.Tensor(config.hidden_size,1))
        self.tang_label = nn.Parameter(torch.Tensor(config.hidden_size,1))
        self.yi_label = nn.Parameter(torch.Tensor(config.hidden_size,1))
        self.lei_label = nn.Parameter(torch.Tensor(config.hidden_size,1))
        self.gan_label = nn.Parameter(torch.Tensor(config.hidden_size,1))
        
        nn.init.uniform_(self.bai_label, -0.1, 0.1)
        nn.init.uniform_(self.qing_label, -0.1, 0.1)
        nn.init.uniform_(self.huang_label, -0.1, 0.1)
        nn.init.uniform_(self.tang_label, -0.1, 0.1)
        nn.init.uniform_(self.yi_label, -0.1, 0.1)
        nn.init.uniform_(self.lei_label, -0.1, 0.1)
        nn.init.uniform_(self.gan_label, -0.1, 0.1)
        
        # 注意softmax的dim
        self.pos_nega_att_softmax = nn.Softmax(dim = 1)
        self.label_att_softmax = nn.Softmax(dim = 0)
        
        self.vec_multi = nn.MultiheadAttention(config.hidden_size, 8, batch_first = True)
        
        # transfomer layer
        trans_layer = torch.nn.TransformerEncoderLayer(config.hidden_size, nhead = 8, batch_first = True, activation = "gelu", dropout = self.dropout_rate, dim_feedforward = 1024)
        self.transformer1 = torch.nn.TransformerEncoder(encoder_layer = trans_layer, num_layers = 5)
        self.transformer2 = torch.nn.TransformerEncoder(encoder_layer = trans_layer, num_layers = 1)

    def forward(self, sym_input_ids, sym_attention_mask,sym_token_type_ids):
             
        # all positive encoding
        
        vec_pool_tuple =  self.all_sym_vec(sym_input_ids, sym_token_type_ids)
        all_organ_vec_bai,all_organ_vec_qing,all_organ_vec_tang,all_organ_vec_yi,all_organ_vec_lei,all_organ_vec_gan = vec_pool_tuple 
        
        logits_bai = self.bai_classifier(all_organ_vec_bai)
        logits_qing = self.qing_classifier(all_organ_vec_qing)
        logits_tang = self.tang_classifier(all_organ_vec_tang)
        logits_yi = self.yi_classifier(all_organ_vec_yi)
        logits_lei = self.lei_classifier(all_organ_vec_lei)
        logits_gan = self.gan_classifier(all_organ_vec_gan)
        
        logits = torch.cat([logits_qing,logits_tang,logits_bai,logits_yi,logits_lei,logits_gan],dim = -1)
                
        return logits
    
    # 获得一个样本的 各个sym表示
    def sample_sym_vec(self, one_sym_input_ids, one_token_type_ids, transformer):
        # 求一个样本的positive_sym_num
        one_sym_input_ids = one_sym_input_ids.cpu().numpy().tolist()
        a = one_token_type_ids.cpu().numpy().tolist()
        g = [a[:1]]
        [g[-1].append(y) if x == y else g.append([y]) for x, y in zip(a[:-1], a[1:])]
        
        # the element of one_all_sym_inputs is a list of inputs_id of one sym
        one_all_sym_inputs = []
        i = 0

        for li_ in g:
            # [cls] [sep]
            one_all_sym_inputs.append([101] + one_sym_input_ids[i : i + len(li_)])
            i += len(li_)
        
        one_all_sym_inputs = one_all_sym_inputs[1:]
        one_all_sym_inputs = [x if len(x) <= 64 else x[:64] for x in one_all_sym_inputs]
        #sym_attention_mask = torch.tensor([[1] * len(x) + [0] * (64 - len(x)) for x in one_all_sym_inputs],dtype = torch.float).cuda(0)
        one_all_sym_inputs = torch.tensor([x  + [0] * (64 - len(x)) for x in one_all_sym_inputs],dtype = torch.long).to(self.device)
        
        one_all_sym_inputs = self.embed(one_all_sym_inputs)
        transformer_result = transformer(one_all_sym_inputs)
        vec_pool = torch.mean(transformer_result, dim = 1)
        return vec_pool
    
    # 获得所有样本的postive表示 
    def all_sym_vec(self, sym_input_ids, sym_token_type_ids):
        all_organ_vec_bai = None
        all_organ_vec_qing = None
        all_organ_vec_tang = None
        all_organ_vec_yi = None
        all_organ_vec_lei = None
        all_organ_vec_gan = None
        
        for i in range(sym_input_ids.shape[0]):
            positive_vec_pool = self.sample_sym_vec(sym_input_ids[i], sym_token_type_ids[i], self.transformer1)
            #negative_vec_pool = self.sample_sym_vec(no_sym_input_ids[i], no_sym_token_type_ids[i], self.transformer1)
            
            #pos_nega_vec = self.pos_nega_attention(positive_vec_pool, negative_vec_pool)[0]
            
            #vec_pool = torch.cat((positive_vec_pool, negative_vec_pool),dim = 0)
            vec_pool = positive_vec_pool.unsqueeze(0)
            vec_pool = self.transformer2(vec_pool).squeeze(0)
            #vec_pool = torch.cat([vec_pool, positive_vec_pool],dim = 0)
            
            organ_ave_bai = self.label_wise_attention(vec_pool, self.bai_label)[0]
            organ_ave_qing = self.label_wise_attention(vec_pool, self.qing_label)[0]
            organ_ave_tang = self.label_wise_attention(vec_pool, self.tang_label)[0]
            organ_ave_yi = self.label_wise_attention(vec_pool, self.yi_label)[0]
            organ_ave_lei = self.label_wise_attention(vec_pool, self.lei_label)[0]
            organ_ave_gan = self.label_wise_attention(vec_pool, self.gan_label)[0]
            
            if all_organ_vec_bai == None:
                all_organ_vec_bai = organ_ave_bai
                all_organ_vec_qing = organ_ave_qing
                all_organ_vec_tang = organ_ave_tang
                all_organ_vec_yi = organ_ave_yi
                all_organ_vec_lei = organ_ave_lei
                all_organ_vec_gan = organ_ave_gan
                
            else:
                all_organ_vec_bai = torch.cat((all_organ_vec_bai,organ_ave_bai),dim = 0)
                all_organ_vec_qing = torch.cat((all_organ_vec_qing,organ_ave_qing),dim = 0)
                all_organ_vec_tang = torch.cat((all_organ_vec_tang,organ_ave_tang),dim = 0)
                all_organ_vec_yi = torch.cat((all_organ_vec_yi,organ_ave_yi),dim = 0)
                all_organ_vec_lei = torch.cat((all_organ_vec_lei,organ_ave_lei),dim = 0)
                all_organ_vec_gan = torch.cat((all_organ_vec_gan,organ_ave_gan),dim = 0)
               
        
        return all_organ_vec_bai,all_organ_vec_qing,all_organ_vec_tang,all_organ_vec_yi,all_organ_vec_lei,all_organ_vec_gan        
    


        
    def label_wise_attention(self, vec_pool, label_embedding):
        att_intern = torch.matmul(vec_pool, label_embedding)
        att_score = self.label_att_softmax(att_intern)
        att_out = torch.matmul(att_score.T, vec_pool)
        
        return att_out, att_score
    
    def pos_nega_attention(self, positive_vec_pool, negative_vec_pool):
        att_intern = torch.matmul(positive_vec_pool, negative_vec_pool.T)
        att_score = self.pos_nega_att_softmax(att_intern)
        att_out = torch.matmul(att_score, negative_vec_pool)
        
        return att_out, att_score
    
#     def attention_net_2(self,organ_ave_pool,w_omega,u_omega):
#         u = torch.tanh(torch.matmul(organ_ave_pool, w_omega))
#         attention = torch.matmul(u, u_omega)
#         att_score = F.softmax(attention, dim=0)
#         scored_x = organ_ave_pool * att_score
#         att_out = torch.sum(scored_x, dim = 0) #加权求和

#         return att_out,att_score