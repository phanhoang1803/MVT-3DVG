#!/usr/bin/env python
# coding: utf-8

import torch
import tqdm
import time
import warnings
import os.path as osp
import torch.nn as nn
from torch import optim
from termcolor import colored

from referit3d.in_out.arguments import parse_arguments
from referit3d.in_out.neural_net_oriented import load_scan_related_data, load_referential_data
from referit3d.in_out.neural_net_oriented import compute_auxiliary_data, trim_scans_per_referit3d_data
from referit3d.in_out.pt_datasets.listening_dataset import make_data_loaders
from referit3d.utils import set_gpu_to_zero_position, create_logger, seed_training_code
from referit3d.utils.tf_visualizer import Visualizer
from referit3d.models.referit3d_net import ReferIt3DNet_transformer
from referit3d.models.referit3d_net_utils import single_epoch_train, evaluate_on_dataset, save_predictions_for_visualization
from referit3d.models.utils import load_state_dicts, save_state_dicts
from referit3d.analysis.deepnet_predictions import analyze_predictions
from transformers import BertTokenizer, BertModel

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_predictions(predictions):
    for pred in predictions:
        scan_id = pred['scan_id']
        utterance = pred['utterance']
        bboxes = pred['bboxes']
        predicted_classes = pred['predicted_classes']

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the bounding boxes
        for i, bbox in enumerate(bboxes):
            class_label = predicted_classes[i]
            ax.scatter(bbox[:, 0], bbox[:, 1], bbox[:, 2], label=f'Class: {class_label}', marker='o')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Scan ID: {scan_id}, Utterance: {utterance}')
        ax.legend()
        plt.show()

def log_train_test_information():
        """Helper logging function.
        Note uses "global" variables defined below.
        """
        logger.info('Epoch:{}'.format(epoch))
        for phase in ['train', 'test']:
            if phase == 'train':
                meters = train_meters
            else:
                meters = test_meters

            info = '{}: Total-Loss {:.4f}, Listening-Acc {:.4f}'.format(phase,
                                                                        meters[phase + '_total_loss'],
                                                                        meters[phase + '_referential_acc'])

            if args.obj_cls_alpha > 0:
                info += ', Object-Clf-Acc: {:.4f}'.format(meters[phase + '_object_cls_acc'])

            if args.lang_cls_alpha > 0:
                info += ', Text-Clf-Acc: {:.4f}'.format(meters[phase + '_txt_cls_acc'])

            logger.info(info)
            logger.info('{}: Epoch-time {:.3f}'.format(phase, timings[phase]))
        logger.info('Best so far {:.3f} (@epoch {})'.format(best_test_acc, best_test_epoch))


if __name__ == '__main__':
    
    # Parse arguments
    args = parse_arguments()
    # Read the scan related information
    all_scans_in_dict, scans_split, class_to_idx = load_scan_related_data(args.scannet_file)
    # Read the linguistic data of ReferIt3D
    referit_data = load_referential_data(args, args.referit3D_file, scans_split)
    # Prepare data & compute auxiliary meta-information.
    all_scans_in_dict = trim_scans_per_referit3d_data(referit_data, all_scans_in_dict)
    mean_rgb, vocab = compute_auxiliary_data(referit_data, all_scans_in_dict, args)
    data_loaders = make_data_loaders(args, referit_data, vocab, class_to_idx, all_scans_in_dict, mean_rgb)
    # Prepare GPU environment
    set_gpu_to_zero_position(args.gpu)  # Pnet++ seems to work only at "gpu:0"

    device = torch.device('cuda')
    seed_training_code(args.random_seed)

    # Losses:
    criteria = dict()
    # Prepare the Listener
    n_classes = len(class_to_idx) - 1  # -1 to ignore the <pad> class
    pad_idx = class_to_idx['pad']
    # Object-type classification
    class_name_list = []
    for cate in class_to_idx:
        class_name_list.append(cate)

    tokenizer = BertTokenizer.from_pretrained(args.bert_pretrain_path)
    class_name_tokens = tokenizer(class_name_list, return_tensors='pt', padding=True)
    for name in class_name_tokens.data:
        class_name_tokens.data[name] = class_name_tokens.data[name].cuda()

    gpu_num = len(args.gpu.strip(',').split(','))

    if args.model == 'referIt3DNet_transformer':
        model = ReferIt3DNet_transformer(args, n_classes, class_name_tokens, ignore_index=pad_idx)
    else:
        assert False

    if gpu_num > 1:
        model = nn.DataParallel(model)
    
    model = model.to(device)
    print(model)
    
    # <1>
    if gpu_num > 1:
        param_list=[
            {'params':model.module.language_encoder.parameters(),'lr':args.init_lr*0.1},
            {'params':model.module.refer_encoder.parameters(), 'lr':args.init_lr*0.1},
            {'params':model.module.object_encoder.parameters(), 'lr':args.init_lr},
            {'params':model.module.obj_feature_mapping.parameters(), 'lr': args.init_lr},
            {'params':model.module.box_feature_mapping.parameters(), 'lr': args.init_lr},
            {'params':model.module.language_clf.parameters(), 'lr': args.init_lr},
            {'params':model.module.object_language_clf.parameters(), 'lr': args.init_lr},
        ]
        if not args.label_lang_sup:
            param_list.append( {'params':model.module.obj_clf.parameters(), 'lr': args.init_lr})
    else:
        param_list=[
            {'params':model.language_encoder.parameters(),'lr':args.init_lr*0.1},
            {'params':model.refer_encoder.parameters(), 'lr':args.init_lr*0.1},
            {'params':model.object_encoder.parameters(), 'lr':args.init_lr},
            {'params':model.obj_feature_mapping.parameters(), 'lr': args.init_lr},
            {'params':model.box_feature_mapping.parameters(), 'lr': args.init_lr},
            {'params':model.language_clf.parameters(), 'lr': args.init_lr},
            {'params':model.object_language_clf.parameters(), 'lr': args.init_lr},
        ]
        if not args.label_lang_sup:
            param_list.append( {'params':model.obj_clf.parameters(), 'lr': args.init_lr})

    optimizer = optim.Adam(param_list,lr=args.init_lr)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,[40, 50, 60, 70, 80, 90], gamma=0.65)

    start_training_epoch = 1
    best_test_acc = -1
    best_test_epoch = -1
    last_test_acc = -1
    last_test_epoch = -1

    if args.resume_path:
        warnings.warn('Resuming assumes that the BEST per-val model is loaded!')
        # perhaps best_test_acc, best_test_epoch, best_test_epoch =  unpickle...
        loaded_epoch = load_state_dicts(args.resume_path, map_location=device, model=model)
        print('Loaded a model stopped at epoch: {}.'.format(loaded_epoch))
        print('Loaded a model that we do NOT plan to fine-tune.')
        load_state_dicts(args.resume_path, optimizer=optimizer, lr_scheduler=lr_scheduler)
        start_training_epoch = loaded_epoch + 1
        start_training_epoch = 0
        best_test_epoch = loaded_epoch
        best_test_acc = 0
        print('Loaded model had {} test-accuracy in the corresponding dataset used when trained.'.format(
            best_test_acc))


    if args.mode == 'vis':
        del referit_data, vocab, class_to_idx, all_scans_in_dict, mean_rgb
        del data_loaders['train']
        res = save_predictions_for_visualization(model, data_loaders['test'], device, channel_last=False)
        visualize_predictions(res)
