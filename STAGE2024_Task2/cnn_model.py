import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from resnet import resnet34, resnet50, resnext50_32x4d


class TriD(nn.Module):
    """TriD.
    Reference:
      Chen et al. Treasure in Distribution: A Domain Randomization based Multi-Source Domain Generalization for 2D Medical Image Segmentation. MICCAI 2023.
    """
    def __init__(self, p=0.5, eps=1e-6, alpha=0.1):
        """
        Args:
          p (float): probability of using TriD.
          eps (float): scaling parameter to avoid numerical issues.
          alpha (float): parameter of the Beta distribution.
        """
        super().__init__()
        self.p = p
        self.eps = eps
        self.beta = torch.distributions.Beta(alpha, alpha)

    def forward(self, x):
        if random.random() > self.p:
            return x

        N, C, H, W = x.shape

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig

        # Sample mu and var from an uniform distribution, i.e., mu ～ U(0.0, 1.0), var ～ U(0.0, 1.0)
        mu_random = torch.empty((N, C, 1, 1), dtype=torch.float32).uniform_(0.0, 1.0).to(x.device)
        var_random = torch.empty((N, C, 1, 1), dtype=torch.float32).uniform_(0.0, 1.0).to(x.device)

        lmda = self.beta.sample((N, C, 1, 1))
        bernoulli = torch.bernoulli(lmda).to(x.device)

        mu_mix = mu_random * bernoulli + mu * (1. - bernoulli)
        sig_mix = var_random * bernoulli + sig * (1. - bernoulli)
        return x_normed * sig_mix + mu_mix


class LearnableSigmoid(nn.Module):
    def __init__(self, ):
        super(LearnableSigmoid, self).__init__()
        self.weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.relu = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.fill_(1.0)

    def forward(self, input):
        return 1. / (1. + torch.exp(-self.relu(self.weight) * input))


class ImagePool(nn.Module):
    def __init__(self, in_ch):
        super(ImagePool, self).__init__()
        self.gpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_ch, in_ch, 1, 1)

    def forward(self, x):
        net = self.gpool(x)
        net = self.conv(net)
        net = F.interpolate(net, size=x.size()[2:], mode="bilinear", align_corners=False)
        return net


class RCAB(nn.Module):
    def __init__(self, ch, reduction=4):
        super(RCAB, self).__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1, groups=ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1, groups=ch)
        self.se = nn.Sequential(
            # nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(ch, ch // reduction, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(ch // reduction, ch, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x
        net = self.conv1(x)
        net = self.conv2(net)
        net = self.se(net)*net
        net = net + identity
        return net


class MSConv2d(nn.Module):
    def __init__(self, ch, groups=4):
        super(MSConv2d, self).__init__()
        assert ch % groups == 0
        group_ch = ch // groups
        self.convs = nn.ModuleList([
            nn.Conv2d(group_ch, group_ch, 1, 1)
        ])
        for i in range(1, groups-1):
            self.convs.append(
                nn.Conv2d(group_ch, group_ch, 3, 1, padding=i, dilation=i, groups=group_ch)
            )
        self.convs.append(ImagePool(group_ch))
        self.activate = nn.GELU()
        self.norm = nn.BatchNorm2d(ch)
        self.groups = groups

    def forward(self, x):
        features = x.chunk(self.groups, dim=1)
        outs = []
        for i in range(len(features)):
            outs.append(self.convs[i](features[i]))
        net = torch.cat(outs, dim=1)
        net = self.norm(net)
        net = self.activate(net)
        return net


class Gate(nn.Module):
    def __init__(self, in_ch, reduction=4):
        super(Gate, self).__init__()
        self.rcab = RCAB(in_ch, reduction)
        self.msconv = MSConv2d(in_ch)

    def forward(self, x):
        net = self.rcab(x)
        net = self.msconv(net)
        net = net + x
        return net


class CrossConv(nn.Module):
    def __init__(self, ch1, ch2, hidden_state, reduction=4):
        super(CrossConv, self).__init__()
        #fusion
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(ch1+ch2, hidden_state, 1, 1),
            nn.BatchNorm2d(hidden_state),
            nn.GELU(),
            Gate(hidden_state, reduction=reduction)
        )
        #cross
        self.linear_cross1 = nn.Sequential(
            nn.Conv2d(hidden_state, ch1, 1, 1),
            nn.GELU()
        )
        self.linear_cross2 = nn.Sequential(
            nn.Conv2d(hidden_state, ch2, 1, 1),
            nn.GELU()
        )
        self.gate1 = nn.Conv2d(ch1, ch1, 1, 1)
        self.gate2 = nn.Conv2d(ch2, ch2, 1, 1)

    def forward(self, fea1, fea2):
        fusion = torch.cat([fea1, fea2], dim=1)
        fusion = self.fusion_conv(fusion)
        cross_1 = self.gate1(self.linear_cross1(fusion)*fea1) + fea1
        cross_2 = self.gate2(self.linear_cross2(fusion)*fea2) + fea2
        return cross_1, cross_2


class Model(nn.Module):
    """
    simply create a 2-branch network, and concat global pooled feature vector.
    each branch = single resnet34
    """
    def __init__(self, type="efficientnet-b3", mixstyle_layer=[]):
        super(Model, self).__init__()
        self.backbone_oct = EfficientNet.from_pretrained(type)
        self.backbone_fundus = EfficientNet.from_pretrained(type)

        if type == "efficientnet-b3":
            self.cross_ids = [6, 12, 20]
            first_chl = 40
            cross1_chl = 48
            cross2_chl = 96
            cross3_chl = 232
            last_chl = 1536
        elif type == "efficientnet-b4":
            self.cross_ids = [8, 16, 24]
            first_chl = 48
            cross1_chl = 56
            cross2_chl = 112
            cross3_chl = 272
            last_chl = 1792
        else:
            assert NotImplementedError, "No such model type:{}".format(type)

        # 在oct_branch更改第一个卷积层通道数
        self.backbone_oct._conv_stem = nn.Conv2d(256, first_chl, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone_oct._fc = nn.Sequential()
        self.process_oct = nn.Sequential(
            nn.Conv2d(last_chl, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
        )
        self.backbone_fundus._fc = nn.Sequential()
        self.process_fundus = nn.Sequential(
            nn.Conv2d(last_chl, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.cross_conv1 = CrossConv(cross1_chl, cross1_chl, cross1_chl, 1)
        self.cross_conv2 = CrossConv(cross2_chl, cross2_chl, cross2_chl, 2)
        self.cross_conv3 = CrossConv(cross3_chl, cross3_chl, cross3_chl, 2)
        self.cross_conv4 = CrossConv(512, 512, 512, 2)

        self.embedding_layer = nn.Linear(9, 128)
        self.down = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(512 * 2, 128), nn.GELU()
        )

        self.linear_info = nn.Sequential(
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 52),
        )

        self.linear_oct = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 52),
        )

        self.linear_fundus = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 52),
        )

        self.linear = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(128 + 128, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 52),
        )

        self.mixstyle_layer = mixstyle_layer
        self.TriD = TriD()

    def extract_features(self, oct, fundus):
        # Stem
        oct_fea = self.backbone_oct._swish(self.backbone_oct._bn0(self.backbone_oct._conv_stem(oct)))
        fundus_fea = self.backbone_fundus._swish(self.backbone_fundus._bn0(self.backbone_fundus._conv_stem(fundus)))
        # Blocks
        for idx, (oct_block, fundus_block) in enumerate(zip(self.backbone_oct._blocks, self.backbone_fundus._blocks)):
            drop_connect_rate = self.backbone_oct._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.backbone_oct._blocks)  # scale drop connect_rate
            oct_fea = oct_block(oct_fea, drop_connect_rate=drop_connect_rate)

            drop_connect_rate = self.backbone_fundus._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.backbone_fundus._blocks)  # scale drop connect_rate
            fundus_fea = fundus_block(fundus_fea, drop_connect_rate=drop_connect_rate)

            if (idx + 1) == self.cross_ids[0]:
                if 'layer1' in self.mixstyle_layer:
                    oct_fea = self.TriD(oct_fea)
                    fundus_fea = self.TriD(fundus_fea)
                oct_fea, fundus_fea = self.cross_conv1(oct_fea, fundus_fea)
            if (idx + 1) == self.cross_ids[1]:
                if 'layer2' in self.mixstyle_layer:
                    oct_fea = self.TriD(oct_fea)
                    fundus_fea = self.TriD(fundus_fea)
                oct_fea, fundus_fea = self.cross_conv2(oct_fea, fundus_fea)
            if (idx + 1) == self.cross_ids[2]:
                oct_fea, fundus_fea = self.cross_conv3(oct_fea, fundus_fea)
        # Head
        oct_fea = self.backbone_oct._swish(self.backbone_oct._bn1(self.backbone_oct._conv_head(oct_fea)))
        fundus_fea = self.backbone_fundus._swish(self.backbone_fundus._bn1(self.backbone_fundus._conv_head(fundus_fea)))
        return oct_fea, fundus_fea

    def forward(self, oct, fundus, info, deep_sup=False):
        # Image Features
        oct_fea, fundus_fea = self.extract_features(oct, fundus)

        oct_fea = self.process_oct(oct_fea)
        fundus_fea = self.process_fundus(fundus_fea)
        oct_fea, fundus_fea = self.cross_conv4(oct_fea, fundus_fea)
        oct_fea, fundus_fea = self.gap(oct_fea).view(oct.size(0), -1), self.gap(fundus_fea).view(fundus.size(0), -1)

        # Information Embeddings
        one_hot = torch.cat((
            info[:, 1].unsqueeze(-1),
            torch.nn.functional.one_hot(info[:, 0].long(), num_classes=2),
            torch.nn.functional.one_hot(info[:, 2].long(), num_classes=2),
            torch.nn.functional.one_hot(info[:, 3].long(), num_classes=4),
                             ), dim=1)
        info_emb = self.embedding_layer(one_hot).view(oct.size(0), -1)

        # Head
        img_emb = self.down(torch.cat((oct_fea, fundus_fea), dim=-1))
        logit = self.linear(torch.cat((img_emb, info_emb), dim=-1))
        if deep_sup:
            logit_info = self.linear_oct(info_emb)
            logit_oct = self.linear_oct(oct_fea)
            logit_fundus = self.linear_fundus(fundus_fea)
            return [logit, logit_info, logit_oct, logit_fundus]
        else:
            return logit


class MixLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.smapeloss = SMAPELoss()
        self.mseloss = CustomMSELoss()

    def forward(self, labels, preds, w=2.):
        return self.smapeloss(labels, preds, w) + self.mseloss(labels, preds, w)


class CustomMSELoss(nn.Module):
    def __init__(self):
        super(CustomMSELoss, self).__init__()

    def forward(self, labels, preds, w):
        mask = torch.ones_like(labels)
        mask[labels <= 0] = w
        loss = torch.mean(mask.to(dtype=torch.float32).to(preds.device) * (preds - labels) ** 2)
        return loss


class R2Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, labels, preds, w):
        mask = torch.ones_like(labels)
        mask[labels <= 0] = w
        return torch.mean(torch.sum(mask.to(dtype=torch.float32).to(preds.device) * (preds - labels) ** 2, dim=1) / torch.sum((labels - labels.mean()) ** 2, dim=1))


class SMAPELoss(nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, labels, preds, w):
        mask = torch.ones_like(labels)
        mask[labels <= 20] = w
        mask[labels <= 0] = w * 2.
        return torch.mean(mask.to(dtype=torch.float32).to(preds.device) * 2 * torch.abs(preds - labels) / (torch.abs(preds) + torch.abs(labels)))


def Smape_(labels, preds):
    return 1 / preds.shape[0] / preds.shape[1] * torch.sum(2 * torch.abs(preds - labels) / (torch.abs(preds) + torch.abs(labels)))

def R2_(labels, preds):
    return torch.mean(1. - torch.sum((preds - labels) ** 2, dim=1) / torch.sum((labels - labels.mean()) ** 2, dim=1))

def Score(smape, R2):
    return 0.5 * (1. / (smape + 0.1)) + 0.5 * (R2 * 10.)


class EpochLR(torch.optim.lr_scheduler._LRScheduler):
    # lr_n = lr_0 * (1 - epoch / epoch_nums)^gamma
    def __init__(self, optimizer, epochs, gamma=0.9, last_epoch=-1):
        self.lr = optimizer.param_groups[0]['lr']
        self.epochs = epochs
        self.gamma = gamma
        super(EpochLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.lr * pow((1. - self.last_epoch / self.epochs), self.gamma)]