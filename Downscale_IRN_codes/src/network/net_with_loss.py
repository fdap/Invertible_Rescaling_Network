# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""define loss function for network"""

import numpy as np

import mindspore as ms
from mindspore.ops import operations as ops
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore import Tensor
from mindspore.common import dtype as mstype
import mindspore.nn as nn

from src.network.util import ReconstructionLoss

class IRN_loss(nn.Cell):
    """the irn network with redefined loss function"""

    def __init__(self, net_G, opt):
        super(IRN_loss, self).__init__()
        self.netG = net_G

        train_opt = opt['train']
        test_opt = opt['test']
        self.opt = opt
        self.train_opt = train_opt
        self.test_opt = test_opt

        self.img_visual = {}

        self.Reconstruction_forw = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_forw'])
        self.Reconstruction_back = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_back'])
        self.ms_sum = ops.ReduceSum()
        self.cat = ops.Concat(1)
        self.reshape = ops.Reshape()
        self.round = ms.ops.Round()
        self.cast = P.Cast()

    def loss_forward(self, out, y, z):
        l_forw_fit = self.train_opt['lambda_fit_forw'] * self.Reconstruction_forw(out, y)

        z = self.reshape(z, (out.shape[0], -1))
        l_forw_ce = self.train_opt['lambda_ce_forw'] * self.ms_sum(z**2) / z.shape[0]

        return l_forw_fit, l_forw_ce

    def test(self,ref_L, real_H):
        output = self.netG(real_H)

        quant = C.clip_by_value(output[:, :3, :, :], 0, 1)
        quant = quant.asnumpy()
        LR = (quant * 255.).round() /255.
        LR = Tensor(LR)

        self.img_visual["GT"] = real_H[0]
        self.img_visual['LR_ref'] = ref_L[0]
        self.img_visual['LR'] = LR[0]


    def construct(self, ref_L, real_H):
        ### forward downscaling
        output = self.netG(x=real_H)
        LR_ref = ref_L

        ##     l_forw_fit  --LR guidance loss
        ##     l_forw_ce   --distribution matching loss
        l_forw_fit, l_forw_ce = self.loss_forward(output[:, :3, :, :], LR_ref, output[:, 3:, :, :])

        ## total loss
        loss = l_forw_fit + l_forw_ce
        return loss











