import torch
import torch.nn as nn
import torch.nn.functional as F


# Define LPD_Net
class LPD_Net(nn.Module):
    def __init__(self, LayerNo):
        super(LPD_Net, self).__init__()
        self.name = "LPD_Net"
        self.LayerNo = LayerNo

        # 卷积层超参数
        self.filter_size = 3
        self.conv_size = 32

        # 两个步长，x: eta_step; y: sigma_step
        self.eta_step = nn.ParameterList()
        self.sigma_step = nn.ParameterList()

        # 阈值参数，soft_thr, soft_a
        self.soft_thr = nn.ParameterList()
        self.soft_a = nn.ParameterList()

        # 激活函数中的参数delta
        self.delta = nn.ParameterList()

        # 一些卷积层        
        self.A2 = nn.ModuleList()
        self.B = nn.ModuleList()

        self.AT2 = nn.ModuleList()
        self.BT = nn.ModuleList()

        # 步长eta_step和sigma_step是不共享的，其余参数：卷积、阈值等是所有层共享的
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

        # 卷积核参数初始化：xavier
        nn.init.xavier_normal_(self.A2[0].weight)
        nn.init.xavier_normal_(self.B[0].weight)
        nn.init.xavier_normal_(self.AT2[0].weight)
        nn.init.xavier_normal_(self.BT[0].weight)

    # 非线性变换中的激活函数
    def activate(self, x, sigma):
        mask1 = (x <= -1*sigma).float()
        mask2 = (torch.abs(x) < sigma).float()
        mask3 = (x >= sigma).float()
        return mask1 * 0. + torch.mul(mask2, x*x/(4*sigma) + x/2 + sigma/4.0) + torch.mul(mask3, x)
        # return F.relu(x)

    # 非线性变换中的激活函数的导函数
    def dif_activate(self, x, sigma):
        mask1 = (x <= -1*sigma).float()
        mask2 = (torch.abs(x) < sigma).float()
        mask3 = (x >= sigma).float()
        return mask1 * 0. + torch.mul(mask2, x/(2*sigma) + 0.5) + torch.mul(mask3, 1.)
        # return F.relu(x)

    def project_sig_y(self, y, thr, a):
        return 2 * thr * (torch.sigmoid(a * y) - 0.5)
    
    def project_relu_y(self, y, thr):
        return torch.mul(torch.sign(y), -1 * F.relu(thr - torch.abs(y)) + thr)

    def forward(self, Phix, Phi, Qinit):
        # batch-size大小 bs
        bs = Phix.size(0)
        # 初始化的x0，y0
        x0 = torch.mm(Phix, torch.transpose(Qinit, 0, 1))
        y0 = torch.zeros(size=[bs, self.conv_size, 33, 33], dtype=torch.float32).to(x0.device)

        PhiTPhi = torch.mm(torch.transpose(Phi, 0, 1), Phi)
        PhiTb = torch.mm(Phix, Phi)

        # x_out, y_out:每层输出的x，y
        x_out = list()
        y_out = list()
        x_out.append(x0)
        y_out.append(y0)
        constraint = list()

        delta = 0.1
        for i in range(self.LayerNo):
            kk = 0

            ##### ======== update y ======== #####
            # 计算更新y的梯度dy
            x_input = x_out[-1].view(-1, 1, 33, 33)

            y_1 = self.A2[0](x_input)
            y_2 = self.activate(y_1, self.delta[i])
            dy = self.B[kk](y_2)

            # 更新y：第(1)步:梯度下降
            y_pred = y_out[-1] + self.sigma_step[i] * dy

            # 更新y：第(2)步:对y进行投影
            y_pred = self.project_sig_y(y_pred, self.soft_thr[i], self.soft_a[i])


            # 存储更新后的y
            y_out.append(y_pred)

            ##### ======== update x ======== #####
            # 计算更新x的梯度dx
            s_1 = self.BT[kk](y_out[-1])

            s_2 = self.A2[kk](x_out[-1].view(-1, 1, 33, 33))
            s_2 = self.dif_activate(s_2, self.delta[i])
            
            s_3 = torch.mul(s_2, s_1)

            s_4 = self.AT2[kk](s_3)
            s_conv = s_4.view(-1, 1089)

            dx = torch.mm(x_out[-1], PhiTPhi) - PhiTb + s_conv

            # 更新x: 梯度下降
            x_pred = x_out[-1] - self.eta_step[i] * dx

            # 存储更新后的x
            x_out.append(x_pred)


        constraint.append(self.A2[0].weight.data - self.AT2[0].weight.data.transpose(0, 1).contiguous())
        constraint.append(self.B[0].weight.data - self.BT[0].weight.data.transpose(0, 1).contiguous())

        return [x_out, constraint]
