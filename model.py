import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform
from torch.autograd import Variable


class SeqDetect(nn.Module):

    def __init__(self, input_size, output_size, base=None, adaptive_pooling=True, nb_steps=10, hidden_size=128, use_cuda=True):
        super().__init__()
        self.base = base
        self.nb_steps = nb_steps
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.adaptive_pooling = adaptive_pooling
        self.rnn = nn.LSTMCell(output_size, hidden_size)
        self.fc_init = nn.Sequential(
            nn.Linear(input_size, hidden_size)
        )
        self.fc_output = nn.Sequential(
            nn.Linear(hidden_size, output_size)
        )
        self.rnn.apply(weights_init)
        self.fc_init.apply(weights_init)
        self.fc_output.apply(weights_init)
        self.use_cuda = use_cuda

    def forward(self, x):
        if self.base:
            f = self.base(x)
            if self.adaptive_pooling:
                f = nn.AdaptiveAvgPool2d((1, 1))(f)
        else:
            f = x
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
        return o

def weights_init(m):
    if hasattr(m, 'weight'):
        if m.weight.size() == 2:
            xavier_uniform(m.weight.data)
