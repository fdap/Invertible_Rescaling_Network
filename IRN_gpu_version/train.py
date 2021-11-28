from gc import enable
import os
import ast
from pickle import TRUE
import cv2
import math
import argparse
import random
import numpy as np
from numpy.lib.function_base import gradient
import time

import mindspore as ms
import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.common import set_seed
from mindspore.train.model import Model
from mindspore.train.callback import TimeMonitor, LossMonitor, CheckpointConfig, ModelCheckpoint, _InternalCallbackParam, RunContext, SummaryCollector
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.model import ParallelMode
from mindspore import Model, load_checkpoint, save_checkpoint, load_param_into_net

import src.options.options as option
import src.utils.util as util
from src.data import create_dataset
from src.optim import warmup_step_lr, warmup_cosine_annealing_lr
from src.optim.adam_clip import AdamClipped
from src.network import create_model, TrainOneStepCell_IRN, IRN_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="srcnn training")
    parser.add_argument('-opt', type=str, help='Path to option YMAL file.')
    parser.add_argument('--device_num', type=int,
                        default=1, help='Device num.')
    parser.add_argument('--device_target', type=str, default='GPU', choices=("GPU"),
                        help="Device target, support GPU.")
    parser.add_argument("--run_distribute", type=ast.literal_eval, default=False,
                        help="Run distribute, default: false.")  # 对字符串进行类型转换的同时兼顾系统的安全考虑

    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    # initialize context
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=args.device_target,
                        save_graphs=False)

    # parallel environment setting
    rank = 0
    if args.run_distribute:
        opt['dist'] = True
        if args.device_target == "Ascend":
            device_id = int(os.getenv('DEVICE_ID'))
            context.set_context(device_id=device_id,
                                enable_auto_mixed_precision=True)
            context.set_auto_parallel_context(device_num=args.device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradient_mean=True)
        elif args.device_target == "GPU":
            context.set_context(device_num=get_group_size(),
                                parallel_mode=ParallelMode.DATA_PARALLEL,
                                gradient_mean=True)
        else:
            raise ValueError("Unsupported device target.")
        init()
        rank = get_rank()
    else:
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')

    context.set_context(max_call_depth=4030)

    # loading options for model
    opt = option.dict_to_nonedict(opt)
    train_opt = opt['train']

    ms.set_seed(train_opt['manual_seed'])

    # create dataset
    dataset_opt = opt['datasets']['train']
    total_epochs = int(opt['train']['epochs'])
    train_dataset = create_dataset("train", dataset_opt, opt['gpu_ids'])
    step_size = train_dataset.get_dataset_size()
    print("Total epoches:{}, Step size:{}".format(total_epochs, step_size))

    # learning rate
    wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
    if train_opt['lr_scheme'] == 'MultiStepLR':
        lr = warmup_step_lr(train_opt['lr_G'],
                            train_opt['lr_steps'],
                            step_size,
                            0,
                            total_epochs,
                            train_opt['lr_gamma'],
                            )
    elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
        lr = warmup_cosine_annealing_lr(train_opt['lr_G'],
                                        train_opt['lr_steps'],
                                        0,
                                        total_epochs,
                                        train_opt['restarts'],
                                        train_opt['eta_min'])
    else:
        raise NotImplementedError(
            'MultiStepLR learning rate scheme is enough.')
    print("Total learning rate:{}".format(lr.shape))

    # define net
    net = create_model(opt)

    # loading resume state if exists
    if opt['path'].get('resume_state', None):
        param_dict = load_checkpoint(opt['path']['resume_state'])
        load_param_into_net(net, param_dict)
        lr = lr[190000:]
        print("load ", str(opt['path']['resume_state']))

    # define network with loss
    loss = IRN_loss(net, opt)

    # warp network with optimizer
    optimizer = AdamClipped(loss.netG.trainable_params(), learning_rate=Tensor(lr),
                            beta1=train_opt['beta1'], beta2=train_opt['beta2'], weight_decay=wd_G)
    print("clipped adam optimizer")

    model = Model(network=loss, optimizer=optimizer)

    ckpt_save_steps = step_size*100
    callbacks = [LossMonitor(25), TimeMonitor(data_size=ckpt_save_steps)]
    config_ck = CheckpointConfig(save_checkpoint_steps=ckpt_save_steps,
                                 keep_checkpoint_max=50)
    save_ckpt_path = os.path.join(
        'ckpt/', 'ckpt_one_step_x4/', time.strftime("%Y-%m-%d %H:%M:%S"+'/', time.localtime()) + '/')
    ckpt_cb = ModelCheckpoint(prefix="irn_onestep",
                              directory=save_ckpt_path, config=config_ck)
    callbacks.append(ckpt_cb)

    model.train(total_epochs, train_dataset,
                callbacks=callbacks, dataset_sink_mode=False)
