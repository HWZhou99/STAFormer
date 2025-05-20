import torch
from torch import nn
from torch.nn import functional as F

class NonLocalBlock1D(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super(NonLocalBlock1D, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = nn.Linear(in_channels, self.inter_channels)

        self.W = nn.Sequential(
            nn.Linear(self.inter_channels, in_channels),
            # nn.BatchNorm1d(in_channels)
        )
        # nn.init.constant_(self.W[1].weight, 0)
        # nn.init.constant_(self.W[1].bias, 0)       

        self.theta = nn.Linear(in_channels, self.inter_channels)
        self.phi = nn.Linear(in_channels, self.inter_channels)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''
        batch_size = x.size(0)

        g_x = self.g(x)
        
        theta_x = self.theta(x)
        phi_x = self.phi(x).transpose(1,2)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        # y = y.permute(0, 2, 1).contiguous()
        # y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


if __name__ == '__main__':
    import torch

    img = torch.zeros(2, 8,768)
    net = NonLocalBlock1D(768)
    out = net(img)
    print(out.size())
        