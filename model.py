import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair
import math


from utils import *


class CLTLayer(nn.Module):
    def __init__(self, in_features, out_features, alpha=10, isinput=False, isoutput=False):
        super(CLTLayer, self).__init__()
        self.n_in = in_features
        self.n_out = out_features
        self.isoutput = isoutput
        self.isinput = isinput
        self.alpha = alpha

        self.Mbias = nn.Parameter(torch.Tensor(out_features))

        self.M = Parameter(torch.Tensor(out_features, in_features))
        self.logS = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.M.size(1))
        self.M.data.normal_(0, stdv)
        self.logS.data.zero_().normal_(-9, 0.001)
        self.Mbias.data.zero_()

    def KL(self):
        logS = self.logS.clamp(-11, 11)
        kl = 0.5 * (self.alpha * (self.M.pow(2) + logS.exp()) - logS).sum()
        return kl

    def cdf(self, x, mu=0., sig=1.):
        return 0.5 * (1 + torch.erf((x - mu) / (sig * math.sqrt(2))))

    def pdf(self, x, mu=0., sig=1.):
        return (1 / (math.sqrt(2 * math.pi) * sig)) * torch.exp(-0.5 * ((x - mu) / sig).pow(2))

    def relu_moments(self, mu, sig):
        alpha = mu / sig
        cdf = self.cdf(alpha)
        pdf = self.pdf(alpha)
        relu_mean = mu * cdf + sig * pdf
        relu_var = (sig.pow(2) + mu.pow(2)) * cdf + mu * sig * pdf - relu_mean.pow(2)
        return relu_mean, relu_var

    def forward(self, mu_h, var_h):
        M = self.M
        var_s = self.logS.clamp(-11,11).exp()

        mu_f = F.linear(mu_h, M, self.Mbias)
        # No input variance
        if self.isinput:
            var_f = F.linear(mu_h**2,var_s)
        else:
            var_f = F.linear(var_h + mu_h.pow(2), var_s) + F.linear(var_h, M.pow(2))

        # compute relu moments if it is not an output layer
        if not self.isoutput:
            return self.relu_moments(mu_f, var_f.sqrt())
        else:
            return mu_f, var_f

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.n_in) + ' -> ' \
               + str(self.n_out) \
               + f', isinput={self.isinput}, isoutput={self.isoutput})'


class ConvCLTLayer(CLTLayer):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=10, stride=1,
                 padding=0, dilation=1, groups=1, isinput=False):
        super(CLTLayer, self).__init__()
        self.n_in = in_channels
        self.n_out = out_channels

        self.isinput = isinput
        self.kernel_size = kernel_size
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.alpha = alpha
        self.normal = True

        self.M = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.logS = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.Mbias = nn.Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        n = self.n_in
        for k in range(1, self.kernel_size):
            n *= k
        self.M.data.normal_(0, 1. / math.sqrt(n))
        self.logS.data.zero_().normal_(-9, 0.001)
        self.Mbias.data.zero_()

    def forward(self, mu_h, var_h):
        var_s = self.logS.clamp(-11, 11).exp()
        mu_f = F.conv2d(mu_h, self.M, self.Mbias, self.stride, self.padding, self.dilation, self.groups)
        if self.isinput:
            var_f = F.conv2d(mu_h ** 2, var_s, None, self.stride, self.padding, self.dilation, self.groups)
        else:
            var_f = F.conv2d(var_h + mu_h.pow(2), var_s, None, self.stride, self.padding, self.dilation,
                             self.groups)
            var_f += F.conv2d(var_h, self.M.pow(2), None, self.stride, self.padding, self.dilation, self.groups)

        return self.relu_moments(mu_f, var_f.sqrt())

    def __repr__(self):
        s = ('{name}({n_in}, {n_out}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        s += ', isinput={isinput}'
        s += ')'

        return s.format(name=self.__class__.__name__, **self.__dict__)


class LeNet5Closed(nn.Module):
    def __init__(self, n_dim=1, n_classes=10, large=False, mode='maxent'):
        super(LeNet5Closed, self).__init__()
        self.varfactor = 1.0
        self.beta = 1.0
        self.mode = mode
        if large:
            self.latdim = 4 * 4 * 192 if n_dim == 1 else 5 * 5 * 192
            self.conv1 = ConvCLTLayer(n_dim, 192, 5, stride=2, isinput=True)
            self.conv2 = ConvCLTLayer(192, 192, 5, stride=2)
            self.dense1 = CLTLayer(self.latdim, 1000)
            self.dense2 = CLTLayer(1000, n_classes, isoutput=True)
        else:
            self.latdim = 4 * 4 * 50 if n_dim == 1 else 5 * 5 * 50
            self.conv1 = ConvCLTLayer(n_dim, 20, 5, stride=2, isinput=True)
            self.conv2 = ConvCLTLayer(20, 50, 5, stride=2)
            self.dense1 = CLTLayer(self.latdim, 500)
            self.dense2 = CLTLayer(500, n_classes, isoutput=True)

    def reset_params(self):
        for l in [self.conv1, self.conv2, self.dense1, self.dense2]:
            l.reset_parameters()

    def forward(self, input):
        mu_h1, var_h1 = self.conv1(input, None)
        mu_h2, var_h2 = self.conv2(mu_h1, var_h1)

        mu_h2 = mu_h2.view(-1, self.latdim)
        var_h2 = var_h2.view(-1, self.latdim)

        mu_h3, var_h3 = self.dense1(mu_h2, var_h2)
        mu_pred, var_pred = self.dense2(mu_h3, var_h3)

        return mu_pred, var_pred

    def loss(self, data, target, N):
        mu_pred, var_pred = self.forward(data)
        KLsum = self.dense1.KL() + self.dense2.KL() + self.conv1.KL() + self.conv2.KL()

        return ProbitLoss_var(mu_pred, var_pred, target) + (KLsum / N)

