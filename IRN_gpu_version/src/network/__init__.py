import logging
from mindspore.nn import optim
from mindspore.ops.composite import clip_ops
import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.ops import operations as ops
from mindspore import dtype as mstype
from mindspore.common import ParameterTuple


import src.network.Invnet as Invnet
from .net_with_loss import IRN_loss


# create invertible rescaling network
def create_model(opt):
    model = opt['model']
    if model == 'IRN':
        m = Invnet.define_G(opt)
    print('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m


class TrainOneStepCell_IRN(nn.TrainOneStepCell):
    """
        Encapsulation class of IRN network training
        Appending an optimizer to the training network after that the construct function can be called to create the backward graph.
    """

    def __init__(self, G, optimizer, sens=1.0):
        super(TrainOneStepCell_IRN, self).__init__(G, optimizer, sens)
        self.optimizer = optimizer
        self.G = G
        self.G.set_grad()
        self.G.set_train(True)
        self.grad = ms.ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.weights = ParameterTuple(optimizer.parameters)
        self.reducer_flag = False
        self.G.add_flags(defer_inline=True)
        self.grad_reducer = F.identity
        self.image_visuals = {}

        self.stack = ms.ops.Stack()
        self.norm = nn.Norm()
        self.mul = ms.ops.Mul()

    # 测试模型输出
    def test(self, ref_L, real_H):
        return self.G.test(ref_L, real_H)

    def construct(self, ref_L, real_H):

        # get the gradient
        loss = self.G(ref_L, real_H)
        sens = ops.Fill()(ops.DType()(loss), ops.Shape()(loss), self.sens)
        grads = self.grad(self.G, self.weights)(ref_L, real_H, sens)

        # clipping gradient norm
        max_norm = 10.0
        total_norm = 0.0
        norm_type = 2.0
        for grad in grads:
            param_norm = self.norm(grad)
            total_norm += param_norm**norm_type
        total_norm = total_norm ** (1. / norm_type)
        clip_coef = max_norm / (total_norm + 1e-6)

        if clip_coef < 1:
            new_grads = ()
            for grad in grads:
                new_grads += (self.mul(grad, clip_coef),)  # 更新梯度
            grads = new_grads

        grads = self.grad_reducer(grads)
        return F.depend(loss, self.optimizer(grads))
