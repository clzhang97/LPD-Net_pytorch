import torch
import torch.nn as nn
import torch.nn.functional as F


# Define LPD_Net
class LPD_Net(nn.Module):
    def __init__(self, LayerNo):
        super(LPD_Net, self).__init__()
        self.name = "LPD_Net"
        self.LayerNo = LayerNo

        self.filter_size = 3
        self.conv_size = 32

        self.eta_step = nn.ParameterList()
        self.sigma_step = nn.ParameterList()

        self.soft_thr = nn.ParameterList()
        self.soft_a = nn.ParameterList()

        self.delta = nn.ParameterList()
      
        self.A2 = nn.ModuleList()
        self.B = nn.ModuleList()

        self.AT2 = nn.ModuleList()
        self.BT = nn.ModuleList()

        for _ in range(self.LayerNo):
            self.eta_step.append(nn.Parameter(torch.Tensor([0.1])))
            self.sigma_step.append(nn.Parameter(torch.Tensor([1])))

            self.soft_thr.append(nn.Parameter(torch.Tensor([0.1])))
            self.soft_a.append(nn.Parameter(torch.Tensor([50])))
            self.delta.append(nn.Parameter(torch.Tensor([0.1])))
                    
        self.A2.append(nn.Conv2d(1, self.conv_size, kernel_size=3, stride=1, padding=1, bias=False))
        self.B.append(nn.Conv2d(self.conv_size, self.conv_size, kernel_size=3, stride=1, padding=1, bias=False))

        self.AT2.append(nn.Conv2d(self.conv_size, 1, kernel_size=3, stride=1, padding=1, bias=False))
        self.BT.append(nn.Conv2d(self.conv_size, self.conv_size, kernel_size=3, stride=1, padding=1, bias=False))

        nn.init.xavier_normal_(self.A2[0].weight)
        nn.init.xavier_normal_(self.B[0].weight)
        nn.init.xavier_normal_(self.AT2[0].weight)
        nn.init.xavier_normal_(self.BT[0].weight)

    def activate(self, x, sigma):
        mask1 = (x <= -1*sigma).float()
        mask2 = (torch.abs(x) < sigma).float()
        mask3 = (x >= sigma).float()
        return mask1 * 0. + torch.mul(mask2, x*x/(4*sigma) + x/2 + sigma/4.0) + torch.mul(mask3, x)

    def dif_activate(self, x, sigma):
        mask1 = (x <= -1*sigma).float()
        mask2 = (torch.abs(x) < sigma).float()
        mask3 = (x >= sigma).float()
        return mask1 * 0. + torch.mul(mask2, x/(2*sigma) + 0.5) + torch.mul(mask3, 1.)

    def project_sig_y(self, y, thr, a):
        return 2 * thr * (torch.sigmoid(a * y) - 0.5)
    
    def project_relu_y(self, y, thr):
        return torch.mul(torch.sign(y), -1 * F.relu(thr - torch.abs(y)) + thr)

    def forward(self, Phix, Phi, Qinit):
        bs = Phix.size(0)
        x0 = torch.mm(Phix, torch.transpose(Qinit, 0, 1))
        y0 = torch.zeros(size=[bs, self.conv_size, 33, 33], dtype=torch.float32).to(x0.device)

        PhiTPhi = torch.mm(torch.transpose(Phi, 0, 1), Phi)
        PhiTb = torch.mm(Phix, Phi)

        x_out = list()
        y_out = list()
        x_out.append(x0)
        y_out.append(y0)
        constraint = list()

        delta = 0.1
        for i in range(self.LayerNo):
            kk = 0
            x_input = x_out[-1].view(-1, 1, 33, 33)

            y_1 = self.A2[0](x_input)
            y_2 = self.activate(y_1, self.delta[i])
            dy = self.B[kk](y_2)

            y_pred = y_out[-1] + self.sigma_step[i] * dy
            y_pred = self.project_sig_y(y_pred, self.soft_thr[i], self.soft_a[i])
            y_out.append(y_pred)

            s_1 = self.BT[kk](y_out[-1])

            s_2 = self.A2[kk](x_out[-1].view(-1, 1, 33, 33))
            s_2 = self.dif_activate(s_2, self.delta[i])
            
            s_3 = torch.mul(s_2, s_1)

            s_4 = self.AT2[kk](s_3)
            s_conv = s_4.view(-1, 1089)

            dx = torch.mm(x_out[-1], PhiTPhi) - PhiTb + s_conv
            x_pred = x_out[-1] - self.eta_step[i] * dx
            x_out.append(x_pred)


        constraint.append(self.A2[0].weight.data - self.AT2[0].weight.data.transpose(0, 1).contiguous())
        constraint.append(self.B[0].weight.data - self.BT[0].weight.data.transpose(0, 1).contiguous())

        return [x_out, constraint]
