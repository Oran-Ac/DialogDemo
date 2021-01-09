from dataloader import *
from header import*
# import torch
# from torch.utils.data import DataLoader
# import os

def load_data_for_post_training(args):
    path = f'data/{args["dataset"]}/{args["mode"]}.txt'
    data = PostTrainingData(path, mode=args['mode'], max_len=args['src_len_size'], lang=args['lang'])
    if args['mode'] == 'train':
        train_sampler = torch.utils.data.distributed.DistributedSampler(data)
        iter_ = DataLoader(
            data,shuffle = False,batch_size = args['batch_size'],collate_fn = data.collate,
            sampler=train_sampler
        )
    else:
        pass
    if not os.path.exists(data.pp_path):
        data.save_pickle()
    args['total_steps'] = len(data) * args['epoch'] / args['batch_size']
    args['bimodel'] = args['model']
    return iter_