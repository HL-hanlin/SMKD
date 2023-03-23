import os
import time
import torch
import random
import argparse
import numpy as np
import torch.optim
import torch.nn as nn
import torch.utils.data.sampler
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.weight_norm import WeightNorm

import feature_loader

class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear( indim, outdim, bias = False)
        self.class_wise_learnable_norm = True  #See the issue#4&8 in the github 
        if self.class_wise_learnable_norm:      
            WeightNorm.apply(self.L, 'weight', dim=0) #split the weight update component to direction and norm      

        if outdim <=200:
            self.scale_factor = 2; #a fixed scale factor to scale the output of cos value into a reasonably large input for softmax
        else:
            self.scale_factor = 10; #in omniglot, a larger scale factor is required to handle >1000 output classes.

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm+ 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized) #matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        scores = self.scale_factor* (cos_dist) 

        return scores

def parse_feature(x,n_support):
    x = Variable(x.cuda())
    z_all = x
    z_support = z_all[:, :n_support]
    z_query = z_all[:, n_support:]
    return z_support, z_query

def cos_sim(features1,features2):
    norm1 = torch.norm(features1, dim=-1).reshape(features1.shape[0], 1)
    norm2 = torch.norm(features2, dim=-1).reshape(1, features2.shape[0])
    end_norm = torch.mm(norm1, norm2)
    cos = torch.mm(features1, features2.T) / end_norm
    return cos

def dis(features1, features2):
    return F.pairwise_distance(features1.unsqueeze(0), features2.unsqueeze(0), p=2)

def feature_evaluation_cos(cl_data_file, n_way=5, n_support=5, n_query=15, adaptation=False):
    class_list = cl_data_file.keys()
    select_class = random.sample(class_list, n_way)
    z_all = []
    for cl in select_class:
        img_feat = cl_data_file[cl]
        perm_ids = np.random.permutation(len(img_feat)).tolist()
        z_all.append([np.squeeze(img_feat[perm_ids[i]]) for i in range(n_support + n_query)])  # stack each batch

    z_all = torch.from_numpy(np.array(z_all))
    z_support, z_query = parse_feature(z_all, n_support)
    z_support_proto = z_support.mean(axis = 1)
    z_query = z_query.contiguous().view(n_way * n_query, -1)
    cos_score = cos_sim(z_support_proto, z_query)
    pred = cos_score.cpu().numpy().argmax(axis=0)
    y = np.repeat(range(n_way), n_query)
    acc = (np.mean(pred == y) * 100)
    return acc

def feature_evaluation_knn(cl_data_file, n_way=5, n_support=5, n_query=15, adaptation=False, top_k = 1):
    class_list = cl_data_file.keys()
    select_class = random.sample(class_list, n_way)
    z_all = []
    for cl in select_class:
        img_feat = cl_data_file[cl]
        perm_ids = np.random.permutation(len(img_feat)).tolist()
        z_all.append([np.squeeze(img_feat[perm_ids[i]]) for i in range(n_support + n_query)])  # stack each batch

    z_all = torch.from_numpy(np.array(z_all))
    z_support, z_query = parse_feature(z_all, n_support)
    
    """
    ####### (1-shot knn is the same as 1-shot prototype) #######
    z_support = z_support.contiguous().view(n_way * n_support, -1)
    z_query = z_query.contiguous().view(n_way * n_query, -1)
    cos_score = cos_sim(z_support, z_query)
    pred = cos_score.cpu().numpy().argmax(axis=0)
    y = np.repeat(range(n_way), n_query)
    acc = (np.mean(pred//n_support == y) * 100)
    """

    ####### knn (5-shot) #######
    z_support = z_support.contiguous().view(n_way * n_support, -1)
    z_query = z_query.contiguous().view(n_way * n_query, -1)
    cos_score = cos_sim(z_support, z_query)

    weights, indices = cos_score.topk(top_k, largest=True, sorted=True, dim = 0)
    weights = weights.cpu().numpy()
    indices = indices.cpu().numpy()
    
    pred_class = (indices//n_way)
    pred_weights = np.zeros(shape=(n_way, n_way * n_query))
    
    for i in range(n_way):
        sum_weights = (1.0 * (pred_class == i) * weights).sum(axis = 0) 
        pred_weights[i] = sum_weights
    pred = np.argmax(pred_weights, axis = 0)
    
    y = np.repeat(range(n_way), n_query)
    acc = (np.mean(pred == y) * 100)
    return acc

def feature_evaluation_classifier(cl_data_file, n_way=5, n_support=5, n_query=15, adaptation=False, loss_type = 'dist', num_epochs=301):
    class_list = cl_data_file.keys()
    select_class = random.sample(class_list, n_way)
    z_all = []
    for cl in select_class:
        img_feat = cl_data_file[cl]
        perm_ids = np.random.permutation(len(img_feat)).tolist()
        z_all.append([np.squeeze(img_feat[perm_ids[i]]) for i in range(n_support + n_query)])  # stack each batch

    z_all = torch.from_numpy(np.array(z_all))
    z_support, z_query = parse_feature(z_all, n_support)
    
    z_support   = z_support.contiguous().view(n_way * n_support, -1 )
    z_query     = z_query.contiguous().view(n_way * n_query, -1 )

    y_support = torch.from_numpy(np.repeat(range( n_way ), n_support ))
    y_support = Variable(y_support.cuda())
    
    feat_dim = z_all.shape[-1]
    
    if loss_type == 'softmax':
        linear_clf = nn.Linear(feat_dim, n_way)
    elif loss_type == 'dist':    
        linear_clf = distLinear(feat_dim, n_way)
    linear_clf = linear_clf.cuda()
    
    set_optimizer = torch.optim.SGD(linear_clf.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)

    loss_function = nn.CrossEntropyLoss()
    loss_function = loss_function.cuda()
    
    batch_size = 4
    support_size = n_way * n_support
    scores_eval = []
    for epoch in range(num_epochs):
        rand_id = np.random.permutation(support_size)
        for i in range(0, support_size , batch_size):
            set_optimizer.zero_grad()
            selected_id = torch.from_numpy( rand_id[i: min(i+batch_size, support_size) ]).cuda()
            z_batch = z_support[selected_id]
            y_batch = y_support[selected_id] 
            scores = linear_clf(z_batch)
            loss = loss_function(scores,y_batch)
            loss.backward()
            set_optimizer.step()
        if epoch %100 ==0 and epoch !=0:
            scores_eval.append(linear_clf(z_query))
    
    acc = []
    for each_score in scores_eval:
        pred = each_score.data.cpu().numpy().argmax(axis = 1)
        y = np.repeat(range( n_way ), n_query )
        acc.append(np.mean(pred == y)*100 )
    
    return acc

def test_methods(args,server,epoch,pretrained_weights,file=None, top_k = 5, loss_type='softmax', evaluation_method='cosine', iter_num=10000):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    n_query = 600 - args.num_shots
    few_shot_params = dict(n_way=args.num_ways, n_support=args.num_shots)

    file_path = os.path.join(pretrained_weights,'{}_224_{}_{}.hdf5'.format(args.partition,epoch, args.checkpoint_key))
    if file is not None:
        file_path = file

    print('testfile:',file_path)
    cl_data_file = feature_loader.init_loader(file_path)
    acc_all = []
    print("evaluating over %d examples" % (n_query))
    print("evaluation method: {}".format(evaluation_method))

    report_freq = 10 if evaluation_method == 'classifier' else 1000
    for i in range(iter_num):
        if evaluation_method == 'cosine':
            acc = feature_evaluation_cos(cl_data_file, n_query=n_query, adaptation=False, **few_shot_params)
        elif evaluation_method == 'knn':    
            acc = feature_evaluation_knn(cl_data_file, n_query=n_query, adaptation=False, top_k = top_k, **few_shot_params)
        elif evaluation_method == 'classifier':
            acc = feature_evaluation_classifier(cl_data_file, n_query=n_query, adaptation=False, **few_shot_params, loss_type = loss_type)
            if type(acc) == list:
                acc = acc[-1]
        
        acc_all.append(acc)
        if i % report_freq == 0:
            print("%d steps reached and the mean acc is %g " % (
                i, np.mean(np.array(acc_all))))
                
    acc_mean1 = np.mean(acc_all)
    acc_std1 = np.std(acc_all)
    print('%d Test Acc at 100= %4.2f%% +- %4.2f%%' % (iter_num, acc_mean1, 1.96 * acc_std1 / np.sqrt(iter_num)))
    print(file_path)
    log_info = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    log_info += '\n Epoch %d %d Shots Test Acc at %d= %4.2f%% +- %4.2f%% %s\n' % (
        epoch, args.num_shots, iter_num, acc_mean1, 1.96 * acc_std1 / np.sqrt(iter_num), args.checkpoint_key)

    with open(os.path.join(pretrained_weights,'{}_log_{}_{}.txt'.format(args.partition,server['dataset'],args.checkpoint_key)), 'a+') as f:
        f.write(log_info)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument('--num_ways', default=5, type=int)
    parser.add_argument('--num_shots', default=5, type=int)
    parser.add_argument('--dataset', default='tiered', type=str)
    parser.add_argument('--seed', default=222, type=int)
    parser.add_argument('--partition', default='test', type=str)
    parser.add_argument('--pretrained_weights', default='/home/heyj/dino/checkpoint_tiered/checkpoint.pth', type=str,
                        help="Path to pretrained weights to evaluate.")
    args = parser.parse_args()
    pretrained_weights = '/home/heyj/dino/checkpoint_tiered/'
    test_methods(args,66,1,pretrained_weights)
