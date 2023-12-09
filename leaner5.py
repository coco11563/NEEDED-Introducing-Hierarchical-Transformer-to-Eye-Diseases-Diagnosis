import argparse
import random
import torch.nn as nn
import os
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader,RandomSampler
from tqdm import tqdm, trange
from transformers import BertTokenizer,AdamW, BertConfig, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup,get_cosine_with_hard_restarts_schedule_with_warmup

from copy import deepcopy
from sklearn import metrics
from sklearn.metrics import classification_report

logger = logging.getLogger(__name__)

from torch.nn import functional as F


def compute_kl_loss(self, p, q, pad_mask=None):
    
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    
    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss


class Focal_loss(nn.Module):    
    def __init__(self, alpha=0.25, gamma=2, num_classes = 5, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)      
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """

        super(Focal_loss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
  
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
   
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算        
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数        
        :param labels:  实际类别. size:[B,N] or [B]        
        :return:
        """        
        # assert preds.dim()==2 and labels.dim()==1        
        preds = preds.view(-1,preds.size(-1))        
        self.alpha = self.alpha.to(preds.device)        
        preds_softmax = F.softmax(preds, dim=1) # 这里并没有直接使用log_softmax, 因为后面会用到softmax的结果(当然你也可以使用log_softmax,然后进行exp操作)        
        preds_logsoft = torch.log(preds_softmax)
        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   # 这部分实现nll_loss ( crossempty = log_softmax + nll )        
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))        
        self.alpha = self.alpha.gather(0,labels.view(-1))        
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        loss = torch.mul(self.alpha, loss.t())        
        if self.size_average:        
            loss = loss.mean()        
        else:            
            loss = loss.sum()        
        return loss

class Learner(object):
    def __init__(self, model,train_config):
        self.model = model
        self.train_dataset = train_config.train_dataset
        self.test_dataset = train_config.test_dataset
        self.dev_dataset = train_config.dev_dataset
        self.device = train_config.device
        
        self.model.to(self.device)
        self.epoch_num = train_config.epoch_num
        self.batch_size = train_config.batch_size
        self.lr = train_config.lr
        self.adam_epsilon = train_config.adam_epsilon
        self.warmup = train_config.warmup
        self.w_decay = train_config.w_decay
        self.labels_num = train_config.labels_num
        self.id_label_dict = train_config.id_label_dict
        self.train_loss = [] # 训练过程中的部分batch loss
        self.max_seq_len = train_config.max_seq_len
        # 
        self.f1_list = []
        self.dev_max_f1 = 0
        self.f1_nochange_num = 0
        self.model_name = train_config.model_name
        self.train_acc = []  # 训练过程中的部分batch loss
        
    def train(self):
        self.model.train()
        
        train_dataloader = DataLoader(self.train_dataset,sampler = RandomSampler(self.train_dataset),
                                      batch_size = self.batch_size, pin_memory = True)
        
        optimizer = AdamW(self.model.parameters(),lr = self.lr,
                    eps = self.adam_epsilon,weight_decay = self.w_decay)
        
        # scheduler = get_linear_schedule_with_warmup(optimizer,
        #            num_warmup_steps = 0.1, num_training_steps = len(train_dataloader) * 20)
        
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps = len(train_dataloader) * 2, num_training_steps = len(train_dataloader) * self.epoch_num)
        
        
        # scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
        #                              num_warmup_steps = len(train_dataloader) * 2, num_training_steps = len(train_dataloader) * self.epoch_num, num_cycles = 2)
        
        logger.info("Start training, the training process takes %d epochs", self.epoch_num)
        criterion = nn.BCEWithLogitsLoss()
        

        for epoch in range(self.epoch_num):
            self.model.train()
            epoch_loss = 0
            epoch_num = 0
            pre_label_arr = None
            true_label_arr = None
            all_logists = None
            for i, batch in tqdm(enumerate(train_dataloader)):
                batch = [one.to(self.device) for one in batch]
                
                inputs = {
                "sym_input_ids" : batch[0],
                "no_sym_input_ids" : batch[1],
                "sym_attention_mask" : batch[2],
                "no_sym_attention_mask" : batch[3],
                "sym_token_type_ids" : batch[4],
                "no_sym_token_type_ids" : batch[5],
                }

                labels = batch[-1]
                
                logits = self.model(**inputs)
                #criterion = Focal_loss(num_classes = self.labels_num)
                #criterion = nn.CrossEntropyLoss()
                                
                loss = criterion(logits,labels.float()).mean()
                
                loss.backward()
                #nn.utils.clip_grad_norm_(self.model.parameters(), 1.0,norm_type = 2)
                optimizer.step()
                scheduler.step()
                self.model.zero_grad()
                
                
                epoch_loss += loss.item()
                    
            
            # print("Train hamming_loss : %.4g" % metrics.hamming_loss(true_label_arr,pre_label_arr)) 
            # print("Train zero_one_loss : %.4g" % metrics.zero_one_loss(true_label_arr,pre_label_arr))
            # print("Train f1_score : %.4g" % now_f1)
            # print("Train micro_f1_score : %.4g" % metrics.f1_score(true_label_arr,pre_label_arr,average='micro'))
            # print("Train macro_roc_auc : %.4g" % metrics.roc_auc_score(true_label_arr, all_logists, average='macro'))
            # t = classification_report(true_label_arr,pre_label_arr, target_names = self.id_label_dict.keys(), zero_division=0) 
            # print(t)
            
                
            # if self.dev(epoch) == 1:
            #     print('Over')
            #     return
        torch.save(self.model.state_dict(),"./model/{}_{}.pth".format(self.model_name_or_path,epoch))

                
                    
    def test(self,model_path = None):
        self.model.eval()
        test_dataloader = DataLoader(self.test_dataset, batch_size = self.batch_size)
        
        logger.info("Start test")
        
        with torch.no_grad():
            pre_label_arr = None
            true_label_arr = None
            all_logists = None
            for i, batch in tqdm(enumerate(test_dataloader)):
                batch = [one.to(self.device) for one in batch]
                inputs = {
                "sym_input_ids" : batch[0],
                "no_sym_input_ids" : batch[1],
                "sym_attention_mask" : batch[2],
                "no_sym_attention_mask" : batch[3],
                "sym_token_type_ids" : batch[4],
                "no_sym_token_type_ids" : batch[5],
                }

                labels = batch[-1]
                logits = self.model(**inputs)
                
                
                if pre_label_arr is None:
                    pre_label_arr = np.where(logits.detach().cpu().numpy() > 0,1,0)
                    true_label_arr = labels.detach().cpu().numpy()
                    all_logists = logits.detach().cpu().numpy()
                else:
                    pre_label_arr = np.append(pre_label_arr, np.where(logits.detach().cpu().numpy() > 0,1,0), axis=0)
                    true_label_arr = np.append(true_label_arr, labels.detach().cpu().numpy(), axis=0)
                    all_logists = np.append(all_logists, logits.detach().cpu().numpy(), axis=0)
                    
            print("Test hamming_loss : %.4g" % metrics.hamming_loss(true_label_arr,pre_label_arr)) 
            print("Test zero_one_loss : %.4g" % metrics.zero_one_loss(true_label_arr,pre_label_arr))
            print("Test f1_score : %.4g" % metrics.f1_score(true_label_arr,pre_label_arr,average='macro'))
            t = classification_report(true_label_arr,pre_label_arr, target_names = self.id_label_dict.keys(), zero_division=0) 
            print("Test micro_f1_score : %.4g" % metrics.f1_score(true_label_arr,pre_label_arr,average='micro'))
            print("Test macro_roc_auc : %.4g" % metrics.roc_auc_score(true_label_arr, all_logists, average='macro'))
            print("Test micro_roc_auc : %.4g" % metrics.roc_auc_score(true_label_arr, all_logists, average='micro'))
            print(t)
            
        return None

    def dev(self, epoch):
        self.model.eval()
        dev_dataloader = DataLoader(self.dev_dataset, batch_size = self.batch_size)
        
        logger.info("Start evaluation")
        
        with torch.no_grad():
            pre_label_arr = None
            true_label_arr = None
            all_logists = None
            for i, batch in tqdm(enumerate(dev_dataloader)):
                batch = [one.to(self.device) for one in batch]
                inputs = {
                "sym_input_ids" : batch[0],
                "no_sym_input_ids" : batch[1],
                "sym_attention_mask" : batch[2],
                "no_sym_attention_mask" : batch[3],
                "sym_token_type_ids" : batch[4],
                "no_sym_token_type_ids" : batch[5],
                }

                labels = batch[-1]
                logits = self.model(**inputs)
                
                
                if pre_label_arr is None:
                    pre_label_arr = np.where(logits.detach().cpu().numpy() > 0,1,0)
                    true_label_arr = labels.detach().cpu().numpy()
                    all_logists = logits.detach().cpu().numpy()
                else:
                    pre_label_arr = np.append(pre_label_arr, np.where(logits.detach().cpu().numpy() > 0,1,0), axis=0)
                    true_label_arr = np.append(true_label_arr, labels.detach().cpu().numpy(), axis=0)
                    all_logists = np.append(all_logists, logits.detach().cpu().numpy(), axis=0)
            
            f1 = metrics.f1_score(true_label_arr,pre_label_arr,average='macro')
            ham_loss = metrics.hamming_loss(true_label_arr,pre_label_arr)
            z_o_loss = metrics.zero_one_loss(true_label_arr,pre_label_arr)
            print("Dev hamming_loss : %.4g" % metrics.hamming_loss(true_label_arr,pre_label_arr)) 
            print("Dev zero_one_loss : %.4g" % metrics.zero_one_loss(true_label_arr,pre_label_arr))
            print("Dev f1_score : %.4g" % f1)
            t = classification_report(true_label_arr,pre_label_arr, target_names = self.id_label_dict.keys(), zero_division=0) 
            print("Dev micro_f1_score : %.4g" % metrics.f1_score(true_label_arr,pre_label_arr,average='micro'))
            print("Dev macro_roc_auc : %.4g" % metrics.roc_auc_score(true_label_arr, all_logists, average='macro'))
            print("Dev micro_roc_auc : %.4g" % metrics.roc_auc_score(true_label_arr, all_logists, average='micro'))
            print(t)
            torch.save(self.model.state_dict(),"./model/{}_{}_epoch{}.pth".format(self.model_name,f1,epoch))

            

            
            
            
        return None   