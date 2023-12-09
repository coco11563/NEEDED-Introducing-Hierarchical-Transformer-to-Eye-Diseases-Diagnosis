# 训练参数
class Train_config(object):
    def __init__(self,train_dataset,test_dataset,dev_dataset,model_name = 'transformer_bingshi_seed{}',epoch_num = 10,batch_size = 32,lr = 2e-5,adam_epsilon = 1e-8,warmup = 0.1,w_decay = 1e-2,max_seq_len = 257,id_label_dict = None,device = 'cuda:2', lr_decay = 0.7):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.dev_dataset = dev_dataset
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.lr = lr
        self.adam_epsilon = adam_epsilon
        self.warmup = warmup
        self.w_decay = w_decay
        self.max_seq_len = max_seq_len
        self.labels_num = len(id_label_dict.keys())
        self.id_label_dict = id_label_dict
        self.device = device
        self.hidden_size = 256
        self.num_labels = len(id_label_dict.keys())
        self.model_name = model_name
        self.lr_decay = lr_decay
        
class Predict_config(object):
    def __init__(self,id_label_dict, device = 'cuda:0'):
        self.device = device
        self.hidden_size = 256
        self.id_label_dict = id_label_dict
        self.num_labels = len(id_label_dict.keys())