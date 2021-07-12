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

import mindspore as ms
import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.common import set_seed
from mindspore.train.model import Model
from mindspore.train.callback import TimeMonitor, LossMonitor, CheckpointConfig, ModelCheckpoint
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.model import ParallelMode
from mindspore import Model,load_checkpoint,save_checkpoint,load_param_into_net


import src.options.options as option
import src.utils.util as util
from src.data import create_dataset
from src.optim import warmup_step_lr,warmup_cosine_annealing_lr
from src.network import create_model,TrainOneStepCell_IRN,IRN_loss



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="srcnn training")
    parser.add_argument('-opt', type=str, help='Path to option YMAL file.')
    parser.add_argument('--device_num', type=int, default=1, help='Device num.')
    parser.add_argument('--device_target', type=str, default='GPU', choices=("GPU"),
                        help="Device target, support GPU.")
    parser.add_argument("--run_distribute", type=ast.literal_eval, default=False,
                        help="Run distribute, default: false.")  # 对字符串进行类型转换的同时兼顾系统的安全考虑

    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    ####  initialize context
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=args.device_target,
                        save_graphs=False)


    ####  parallel environment setting                    
    rank = 0
    if args.run_distribute:
        opt['dist'] = True
        if args.device_target == "Ascend":
            device_id = int(os.getenv('DEVICE_ID'))
            context.set_context(device_id=device_id,
                                enable_auto_mixed_precision=True)
            context.set_auto_parallel_context(device_num = args.device_num,
                                              parallel_mode = ParallelMode.DATA_PARALLEL,
                                              gradient_mean = True)
        elif args.device_target == "GPU":
            context.set_context(device_num = get_group_size(),
                                parallel_mode = ParallelMode.DATA_PARALLEL,
                                gradient_mean = True)
        else:
            raise ValueError("Unsupported device target.")
        init()
        rank = get_rank()
    else:
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')


    #### loading options for model
    opt = option.dict_to_nonedict(opt)
    train_opt = opt['train']
    test_opt = opt['test']

    ####  create dataset
    dataset_opt = opt['datasets']['train']
    train_dataset = create_dataset("train",dataset_opt,opt['gpu_ids'])

    step_size = train_dataset.get_dataset_size()
    total_iters = int(opt['train']['niter'])
    total_epochs = int(math.ceil(total_iters / step_size))
    print("Total epoches : {} , Step size : {} , total iters : {}".format(total_epochs,step_size,total_iters))


    #### learning rate 
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
        raise NotImplementedError('MultiStepLR learning rate scheme is enough.')
    print(lr.shape)

    #### define net    
    net = create_model(opt)

    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        param_dict = load_checkpoint(opt['path']['resume_state'])
        load_param_into_net(net, param_dict)

    #### define network with loss
    loss = IRN_loss(net,opt) 

    #### warp network with optimizer
    optimizer = nn.Adam(loss.netG.trainable_params(), learning_rate=Tensor(lr),
                 beta1=train_opt['beta1'], beta2=train_opt['beta2'], weight_decay=wd_G)
    print(len(loss.netG.trainable_params()))

    model = TrainOneStepCell_IRN(loss, optimizer)

    #### 1. Warp the whole network as a Model
    irn_model = Model(network=loss,optimizer=optimizer)
    

    ### 2. define callbacks
    callbacks = [LossMonitor(), TimeMonitor(data_size=step_size)]
    config_ck = CheckpointConfig(save_checkpoint_steps=step_size,
                                 keep_checkpoint_max=30)
    save_ckpt_path = os.path.join('ckpt/', 'ckpt_' + str(rank) + '/')
    ckpt_cb = ModelCheckpoint(prefix="irn", directory=save_ckpt_path, config=config_ck)
    callbacks.append(ckpt_cb)

    ## 注释语句
    # initial_param =irn_model._network.netG.trainable_params()
    # i_param = []
    # for i in range(len(initial_param)):
    #     i_param.append(initial_param[i].asnumpy().copy())
    # i_param = np.array(i_param.copy())


    ### 3. train 
    total_epochs = 1
    irn_model.train(total_epochs, train_dataset, callbacks=callbacks, dataset_sink_mode=False)

    ## 注释语句，用于判断训练前后模型参数是否发生变化
    # new_params = irn_model._network.netG.trainable_params()
    # for j in range(len(initial_param)):
    #     if (i_param[j] == new_params[j].asnumpy()).all():
    #         print(initial_param[j], "not changed at all")
    #     else:
    #         print(initial_param[j], "changed")


    


    ##########################
    #### 单批次进行训练       ##
    ##########################

    ###  1. get the randomly first batch
    # dataset_iter = train_dataset.create_dict_iterator()
    # data = next(dataset_iter)
    # lq = ms.Tensor(data["low_quality"],ms.float32)
    # gt = ms.Tensor(data["ground_truth"],ms.float32)   
    # lq_img = util.tensor2img(lq[0])
    # gt_img = util.tensor2img(gt[0])
    # cv2.imwrite("/home/nonroot/IRN_md_version/irn_md_codes/fig/lq.jpg",lq_img)   
    # cv2.imwrite("/home/nonroot/IRN_md_version/irn_md_codes/fig/gt.jpg",gt_img) 
    # print(lq.shape,gt.shape,"save pics successfully")  

    ### 2. set train
    # model.set_train(True)

    ### 3. train
    # for i in range(total_epochs):
    #     loss1 = model(lq,gt)  
    #     print("step :{}, loss is {}".format(i,loss1))
    #     if i==500:
    #         break    ## 节约时间仅跑一轮

    ### 3. test the model for output visualization
    # images = model.test(lq,gt)
    # lq_img = util.tensor2img(images["LR"])
    # cv2.imwrite("./fig/out_lq.jpg",lq_img)
