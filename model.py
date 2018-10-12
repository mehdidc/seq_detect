import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform
from torch.autograd import Variable


class SeqDetect(nn.Module):

    def __init__(self, base=None, input_size=224, output_size=4, nb_feature_maps=512, 
                 upsize=128, upscale=5, adaptive_pooling=True, nb_steps=10, hidden_size=128, use_cuda=True):
        super().__init__()
        self.base = base
        self.nb_steps = nb_steps
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.adaptive_pooling = adaptive_pooling
        self.rnn = nn.LSTMCell(output_size, hidden_size)
        l = []
        l.append(nn.ConvTranspose2d(nb_feature_maps, upsize , 4, 2, 1, bias=False))
        l.append(nn.BatchNorm2d(upsize))
        l.append(nn.ReLU(True))
        for i in range(upscale - 2):
            l.append(nn.ConvTranspose2d(upsize, upsize, 4, 2, 1, bias=False))
            l.append(nn.BatchNorm2d(upsize))
            l.append(nn.ReLU(True))
        l.append(nn.ConvTranspose2d(upsize, 1, 4, 2, 1, bias=True))
        self.upconv = nn.Sequential(*l)
        self.fc_init = nn.Sequential(
            #nn.Linear(input_size * input_size, hidden_size)
            nn.Linear(nb_feature_maps, hidden_size)
        )
        self.fc_output = nn.Sequential(
            nn.Linear(hidden_size, output_size)
        )
        self.rnn.apply(weights_init)
        self.fc_init.apply(weights_init)
        self.fc_output.apply(weights_init)
        self.upconv.apply(weights_init)
        self.use_cuda = use_cuda

    def forward(self, x):
        h = self.base(x)
        m = self.upconv(h)
        f = nn.AdaptiveAvgPool2d((1, 1))(h)
        f = f.view(f.size(0), -1)
        o = torch.zeros(f.size(0), self.output_size)
        o = Variable(o)
        if self.use_cuda:
            o = o.cuda()
        hx = self.fc_init(f)
        cx = hx
        outs = []
        for t in range(self.nb_steps):
            hx, cx = self.rnn(o, (hx, cx))
            o = self.fc_output(hx)
            outs.append(o)
        o = torch.stack(outs, 0)
        o = o.permute(1, 0, 2)
        return o, m


def weights_init(m):
    if hasattr(m, 'weight'):
        if m.weight.size() == 2:
            xavier_uniform(m.weight.data)
