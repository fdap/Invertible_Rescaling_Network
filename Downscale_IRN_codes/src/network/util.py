import mindspore
import mindspore.nn as nn
import mindspore.common.initializer as init
from mindspore.ops import operations as ops
from mindspore.ops import functional as F


class ReconstructionLoss(nn.Cell):
    def __init__(self, losstype='l2', eps=1e-6):
        super(ReconstructionLoss, self).__init__()
        self.losstype = losstype
        self.eps = eps

        self.mean = ops.ReduceMean()
        self.sum = ops.ReduceSum()
        self.sqrt = ops.Sqrt()

    def construct(self, x, target):
        if self.losstype == 'l2':
            return self.mean(self.sum((x - target)**2, (1, 2, 3)))
        elif self.losstype == 'l1':
            diff = x - target
            return self.mean(self.sum(self.sqrt(diff * diff + self.eps), (1, 2, 3)))
        else:
            print("reconstruction loss type error!")
            return 0



def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for _,m in net.cells_and_names():   # 遍历网络中的每一个Cell模块
            if isinstance(m, nn.Conv2d):
                m.weight.set_data(init.initializer(init.HeNormal(negative_slope=0,mode='fan_in'),m.weight.shape,m.weight.dtype))
                m.weight.set_data(m.weight.data * scale)  # for residual block
                if m.bias is not None:
                    m.bias.set_data(init.initializer(init.Zero(),m.bias.shape,m.bias.dtype))
                    m.bias.requires_grad=True
            elif isinstance(m, nn.Dense):
                m.weight.set_data(init.initializer(init.HeNormal(negative_slope=0,mode='fan_in'),m.weight.shape,m.weight.dtype))
                m.weight.set_data(m.weight.data * scale)  # for residual block
                if m.bias is not None:
                    m.bias.set_data(init.initializer(init.Zero(),m.bias.shape,m.bias.dtype))
                    m.bias.requires_grad=True
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.set_data(init.initializer(init.Constant(1),m.weight.shape,m.weight.dtype))
                m.bias.set_data(init.initializer(init.Zero(),m.bias.shape,m.bias.dtype))
                m.bias.requires_grad=True
                


def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for _,m in net.cells_and_names():
            if isinstance(m, nn.Conv2d):
                m.weight.set_data(init.initializer(init.XavierUniform(),m.weight.shape,m.weight.dtype))
                m.weight.set_data(m.weight.data * scale)  # for residual block
                if m.bias is not None:
                    m.bias.set_data(init.initializer(init.Zero(),m.bias.shape,m.bias.dtype))
                    m.bias.requires_grad=True
            elif isinstance(m, nn.Dense):
                m.weight.set_data(init.initializer(init.XavierUniform(),m.weight.shape,m.weight.dtype))
                m.weight.set_data(m.weight.data * scale)
                if m.bias is not None:
                    m.bias.set_data(init.initializer(init.Zero(),m.bias.shape,m.bias.dtype))
                    m.bias.requires_grad=True
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.set_data(init.initializer(init.Constant(1),m.weight.shape,m.weight.dtype))
                m.bias.set_data(init.initializer(init.Zero(),m.bias.shape,m.bias.dtype))
                m.bias.requires_grad=True

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.SequentialCell(layers)


class ResidualBlock_noBN(nn.Cell):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, "pad", 1, has_bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, "pad", 1, has_bias=True)

        self.relu = nn.Relu()

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return identity + out


