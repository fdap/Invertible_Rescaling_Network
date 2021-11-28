from gc import enable
import os
import ast
from pickle import TRUE
import cv2
import math
import time
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
from mindspore import Model, load_checkpoint, save_checkpoint, load_param_into_net


import src.options.options as option
import src.utils.util as util
from src.data.util import bgr2ycbcr
from src.data import create_dataset
from src.optim import warmup_step_lr, warmup_cosine_annealing_lr
from src.network import create_model, TrainOneStepCell_IRN, IRN_loss


if __name__ == '__main__':
    begin = time.time()
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
    opt = option.parse(args.opt, is_train=False)

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

    # loading options for model
    opt = option.dict_to_nonedict(opt)

    # create dataset
    dataset_opt = opt['datasets']['test_5']
    val_dataset = create_dataset("val", dataset_opt, opt['gpu_ids'])

    step_size = val_dataset.get_dataset_size()
    print("Step size : {}".format(step_size))

    # define net
    net = create_model(opt)

    # loading resume state if exists
    if opt['path'].get('pretrain_model_G', None):
        param_dict = load_checkpoint(opt['path']['pretrain_model_G'])
        load_param_into_net(net, param_dict)
        print("saved model restore! " + str(opt['path']['pretrain_model_G']))

    # define network with loss
    loss = IRN_loss(net, opt)

    val_iter = val_dataset.create_dict_iterator()

    idx = 0
    test_hr_psnr = []
    test_hr_ssim = []
    test_y_hr_psnr = []
    test_y_hr_ssim = []

    test_lr_psnr = []
    test_y_lr_psnr = []
    test_lr_ssim = []
    test_y_lr_ssim = []

    for _ in range(val_dataset.get_dataset_size()):
        idx += 1
        val = next(val_iter)
        lq = ms.Tensor(val["low_quality"], ms.float32)
        gt = ms.Tensor(val["ground_truth"], ms.float32)
        images = loss.test(lq, gt)
        sr_img = util.tensor2img(images[3])
        gt_img = util.tensor2img(images[0])
        gt_lr_img = util.tensor2img(images[1])
        lq_img = util.tensor2img(images[2])
        if idx == 1:
            util.save_img(sr_img, "./fig/val_fig/sr_0.png")
            util.save_img(gt_lr_img, "./fig/val_fig/gt_lr_0.png")
            util.save_img(gt_img, "./fig/val_fig/gt_0.png")
            util.save_img(lq_img, "./fig/val_fig/lq_0.png")
        gt_img = gt_img / 255.
        sr_img = sr_img / 255.

        gt_lr_img = gt_lr_img / 255.
        lq_img = lq_img / 255.

        crop_size = opt['scale']
        cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, :]
        cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, :]
        psnr = util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
        ssim = util.calculate_ssim(cropped_sr_img * 255, cropped_gt_img * 255)
        test_hr_psnr.append(psnr)
        test_hr_ssim.append(ssim)

        lr_psnr = util.calculate_psnr(gt_lr_img * 255, lq_img * 255)
        lr_ssim = util.calculate_ssim(gt_lr_img * 255, lq_img * 255)
        test_lr_psnr.append(lr_psnr)
        test_lr_ssim.append(lr_ssim)

        avg_PSNR = sum(test_hr_psnr) / len(test_hr_psnr)
        avg_SSIM = sum(test_hr_ssim) / len(test_hr_ssim)
        lr_avg_PSNR = sum(test_lr_psnr) / len(test_lr_psnr)
        lr_avg_SSIM = sum(test_lr_ssim) / len(test_lr_ssim)

        if gt_img.shape[2] == 3:  # RGB image
            sr_img_y = bgr2ycbcr(sr_img, only_y=True)
            gt_img_y = bgr2ycbcr(gt_img, only_y=True)

            cropped_sr_img_y = sr_img_y[crop_size:-
                                        crop_size, crop_size:-crop_size]
            cropped_gt_img_y = gt_img_y[crop_size:-
                                        crop_size, crop_size:-crop_size]

            psnr_y = util.calculate_psnr(
                cropped_sr_img_y * 255, cropped_gt_img_y * 255)
            ssim_y = util.calculate_ssim(
                cropped_sr_img_y * 255, cropped_gt_img_y * 255)
            test_y_hr_psnr.append(psnr_y)
            test_y_hr_ssim.append(ssim_y)

            lr_img_y = bgr2ycbcr(lq_img, only_y=True)
            lrgt_img_y = bgr2ycbcr(gt_lr_img, only_y=True)
            psnr_y_lr = util.calculate_psnr(lr_img_y * 255, lrgt_img_y * 255)
            ssim_y_lr = util.calculate_ssim(lr_img_y * 255, lrgt_img_y * 255)
            test_y_lr_psnr.append(psnr_y_lr)
            test_y_lr_ssim.append(ssim_y_lr)

            avg_PSNR_y = sum(test_y_hr_psnr) / len(test_y_hr_psnr)
            avg_SSIM_y = sum(test_y_hr_ssim) / len(test_y_hr_ssim)
            lr_avg_PSNR_y = sum(test_y_lr_psnr) / len(test_y_lr_psnr)
            lr_avg_SSIM_y = sum(test_y_lr_ssim) / len(test_y_lr_ssim)

            print('{:4d} - PSNR: {:.6f} dB; SSIM: {:.6f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}. LR PSNR: {:.6f} dB; SSIM: {:.6f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.'.
                  format(idx, psnr, ssim, psnr_y, ssim_y, lr_psnr, lr_ssim, psnr_y_lr, ssim_y_lr))
            print('      - avg PSNR: {:.6f} dB; avg SSIM: {:.6f}; avg lr PSNR: {:.6f}; avg lr SSIM: {:.6f}.'.format(
                avg_PSNR, avg_SSIM, lr_avg_PSNR, lr_avg_SSIM))
            print('      - avg PSNR Y: {:.6f} dB; avg SSIM Y: {:.6f}; avg lr PSNR_Y: {:.6f}; avg lr SSIM_Y: {:.6f}.'.format(
                avg_PSNR_y, avg_SSIM_y, lr_avg_PSNR_y, lr_avg_SSIM_y))
        else:
            print('{:4d} - PSNR: {:.6f} dB; SSIM: {:.6f}. LR PSNR: {:.6f} dB; SSIM: {:.6f}.'.
                  format(idx, psnr, ssim, lr_psnr, lr_ssim))
            print('      - avg PSNR: {:.6f} dB; avg SSIM: {:.6f}; avg lr PSNR: {:.6f}; avg lr SSIM: {:.6f}.'.format(
                avg_PSNR, avg_SSIM, lr_avg_PSNR, lr_avg_SSIM))

    print("eval time: ", time.time() - begin)
