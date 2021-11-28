import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import functional as F


class AdamClipped(nn.Adam):

    def __init__(self, params, learning_rate, beta1, beta2, weight_decay, max_norm=10.0, norm_type=2.0):
        super().__init__(params, learning_rate=learning_rate,
                         beta1=beta1, beta2=beta2, weight_decay=weight_decay)
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.norm = nn.Norm()
        self.mul = ops.Mul()

    def construct(self, gradients):
        total_norm = 0.0
        for grad in gradients:
            total_norm += self.norm(grad) ** self.norm_type
        total_norm = total_norm ** (1. / self.norm_type)
        clip_coef = self.max_norm / (total_norm + 1e-6)

        if clip_coef < 1:
            gradients = self.map_(F.partial(self.mul, clip_coef), gradients)
        return super().construct(gradients)
