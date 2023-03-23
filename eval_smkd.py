import os
import sys
import h5py
import torch
import argparse
from torch import nn
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms

import utils
import models as mymodels
from fseval import test_methods

server_dict = {
    'mini':{
        'dataset': 'mini',
        'dataset_name': 'mini-imagenet',
        'data_path': '/home/hanlin/FSDatasets/mini-imagenet-480/'
        },  
    'fs':{
        'dataset': 'fs',
        'dataset_name': 'cifar-fs',
        'data_path': '/home/hanlin/FSDatasets/cifar-fs-84/'
        },
    'fc100':{
        'dataset': 'fc100',
        'dataset_name': 'FC100',
        'data_path': '/home/hanlin/FSDatasets/FC100-84/'
        },
    'tiered':{
        'dataset': 'tiered',
        'dataset_name': 'tiered-imagenet',
        'data_path': '/home/hanlin/FSDatasets/tiered-imagenet-tools-master/tiered_imagenet-480/'
        },
}

def eval_valtest(args):
    
    print("#################### args.weighted_avgpool_patchtokens: {}, args.avgpool_patchtokens: {}, args.cls_tokens: {} ##################" \
          .format(args.weighted_avgpool_patchtokens, args.avgpool_patchtokens, args.cls_tokens))
        
    server = server_dict[args.server]
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    dataset_name = server['dataset_name']
    if 'mini' in dataset_name or 'tiered' in dataset_name:
        mean = tuple([0.485, 0.456, 0.406])
        std = tuple([0.229, 0.224, 0.225])
        img_size = args.img_size_imagenet 
        img_resize = args.img_resize_imagenet
    elif 'cifar' in dataset_name or 'FC100' in dataset_name:
        mean = tuple([x/255.0 for x in [129.37731888,  124.10583864, 112.47758569]])
        std = tuple([x/255.0 for x in [68.20947949,  65.43124043,  70.45866994]])  
        img_size = args.img_size_cifar
        img_resize = args.img_resize_cifar
        
    val_transform = pth_transforms.Compose([
        pth_transforms.Resize(int(img_size), interpolation=3),
        pth_transforms.CenterCrop(int(img_resize)),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize(mean, std),
    ])
    
    dataset_test = datasets.ImageFolder(os.path.join(server['data_path'], args.partition), transform=val_transform)
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Data loaded with {len(dataset_test)} test imgs.")

    # ============ building network ... ============
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in mymodels.__dict__.keys():
        model = mymodels.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    else:
        print(f"Unknow architecture: {args.arch}")
        sys.exit(1)
    model.cuda()
    model.eval()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    checkdir = os.listdir(server['ckp_path'])
    checkdir.sort()
    checkdir = [checkdir[i] for i in range(len(checkdir)) if args.ckpt_filename in checkdir[i]]
    for i in range(len(checkdir)):
        if str(args.epochs) in checkdir[i]:
            if args.epochs != -1:
                checkdir =  checkdir[i:] + checkdir[0:1]
            else:
                checkdir = checkdir[0:1] + checkdir[i:]
            break

    for i in range(len(checkdir)):
        print(f"Evaluating pretrained weight in {checkdir[i]}")
        if '.pth' in checkdir[i]:
            args.pretrained_weights = os.path.join(server['ckp_path'],checkdir[i])
            
            if not checkdir[i][-8:-4].isdigit():
                epoch = int(torch.load(args.pretrained_weights)['epoch']) - 1
            else:
                epoch = int(checkdir[i][-8:-4])

            outfile = os.path.join(args.output_dir,'{}_224_epoch{}_{}_{}shot.hdf5'.format(args.partition,epoch, args.checkpoint_key, args.num_shots))
            if not os.path.isfile(outfile) or args.isfile == 1:
                utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch,
                                              args.patch_size)
                if args.save == 1:
                    save_features(model,server['dataset'], test_loader, args.n, 
                                  args.cls_tokens, args.avgpool_patchtokens, args.weighted_avgpool_patchtokens, 
                                  args.p_scale, args.wp_scale, 
                                  epoch, server['ckp_path'],outfile)

            test_methods(args,server,epoch,server['ckp_path'],outfile, top_k = args.top_k, loss_type=args.loss_type, evaluation_method=args.evaluation_method)
        if int(args.epochs) == -1:
            return

def save_features(model,dataset,loader, n, clstokens, avgpool, weightedavgpool, p_scale, wp_scale, epochs, pretrained_weights,outfile):

    return_attention = True if weightedavgpool else False 

    f = h5py.File(outfile, 'w')
    max_count = len(loader) * loader.batch_size
    print(max_count)
    all_labels = f.create_dataset('all_labels', (max_count,), dtype='i')
    all_feats = None
    
    count = 0
    for i, (inp, target) in enumerate(loader):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        # forward
        with torch.no_grad():
            if "vit" in args.arch:
                intermediate_output, attns = model.get_intermediate_layers(inp, n, return_attention=return_attention)
                if clstokens:
                    output = [x[:,0] for x in intermediate_output]
                else:
                    output = []
                if avgpool:
                    for layer in range(n):
                        output.append(torch.mean(intermediate_output[layer][:, 1:], dim=1) * p_scale) 
                if attns is not None:
                    if weightedavgpool:
                        for layer in range(n):
                            patch_tokens = intermediate_output[layer][:, 1:] #[100, 196, 384]
                            attn_weights = attns[layer].mean(axis = 1)[:, 0, 1:] # [100, 196]
                            attn_weights = attn_weights / attn_weights.sum(axis = -1, keepdims = True)
                            weighted_patch_tokens = patch_tokens * attn_weights.unsqueeze(-1)
                            output.append(torch.sum(weighted_patch_tokens, dim=1) * wp_scale) 
                
                output = [nn.functional.normalize(x, dim=-1) for x in output]
                output = torch.cat(output, dim=-1)

        if i % 10 == 0:
            print('{:d}/{:d}'.format(i, len(loader)))
        if all_feats is None:
            all_feats = f.create_dataset('all_feats', [max_count] + list(output.size()[1:]), dtype='f')
        all_feats[count:count + output.size(0)] = output.data.cpu().numpy()
        all_labels[count:count + output.size(0)] = target.cpu().numpy()

        count = count + output.size(0)

    count_var = f.create_dataset('count', (1,), dtype='i')
    count_var[0] = count

    f.close()
    print(outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation SMKD')
    parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""")
    parser.add_argument('--weighted_avgpool_patchtokens', default=True, type=utils.bool_flag,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""")
    parser.add_argument('--cls_tokens', default=True, type=utils.bool_flag,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""")            
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')

    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')

    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""")
    parser.add_argument('--batch_size_per_gpu', default=300, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--output_dir', default=".", help='Path to save logs and checkpoints')
    parser.add_argument('--num_labels', default=1000, type=int, help='Number of labels for linear classifier') # this is not useful since we have few shot below
    
    # few-shot args
    parser.add_argument('--num_ways', default=5, type=int)
    parser.add_argument('--num_shots', default=1, type=int)
    parser.add_argument('--seed', default=0, type=int)
    
    # evaluation args
    parser.add_argument('--partition', default='test', type=str) 
    parser.add_argument('--epochs', default='-1', type=str, help='Number of epochs of training.')
    parser.add_argument('--save', default=1, type=int)
    parser.add_argument('--isfile', default=-1, type=int)
    parser.add_argument('--server', default='mini', type=str,
                        help='mini / tiered / fs / fc100')
    parser.add_argument('--n',type = int, default=1)
    parser.add_argument('--both',default=1, type=int)
    parser.add_argument('--ckp_path',default='',type=str,
                        help='path to the checkpoint of hct')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--ckpt_filename', default='', type=str, help="checkpoint0020.pth")
    
    parser.add_argument("--wp_scale", default=1, type=float)
    parser.add_argument("--p_scale", default=1, type=float)
    parser.add_argument('--dataset_name', default='', type=str, help="dataset name.")    
    
    parser.add_argument('--knn', default=5, type=int)
    parser.add_argument('--top_k', default=1, type=int)    
    parser.add_argument('--evaluation_method', default='cosine', type=str, 
                        choices=['cosine', 'knn', 'classifier']) # 
    parser.add_argument('--img_size_imagenet', type=int, default=360)
    parser.add_argument('--img_resize_imagenet', type=int, default=320)
    parser.add_argument('--img_size_cifar', type=int, default=256)
    parser.add_argument('--img_resize_cifar', type=int, default=224)
    parser.add_argument('--iter_num', type=int, default=10000)
    
    parser.add_argument('--loss_type', choices=['softmax', 'dist'],
                        default='dist', type=str)    
        
    args = parser.parse_args()
    
    # setup ckp_path
    server_dict[args.server]['ckp_path'] = args.ckp_path
   
    eval_valtest(args)

