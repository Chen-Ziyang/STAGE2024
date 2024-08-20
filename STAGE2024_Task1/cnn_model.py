import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from resnet import resnet18, resnet34, resnet50
from einops import rearrange


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


class CrossAttention(nn.Module):
    def __init__(self, in_channels, emb_dim):
        super(CrossAttention, self).__init__()
        self.emb_dim = emb_dim
        self.scale = emb_dim ** -0.5

        self.proj_in1 = nn.Conv2d(in_channels, emb_dim, kernel_size=1, stride=1, padding=0)
        self.proj_in2 = nn.Conv2d(in_channels, emb_dim, kernel_size=1, stride=1, padding=0)

        self.Wq = nn.Linear(emb_dim, emb_dim)
        self.Wk = nn.Linear(emb_dim, emb_dim)
        self.Wv = nn.Linear(emb_dim, emb_dim)

        self.proj_out = nn.Conv2d(emb_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, attn):
        b, c, h, w = x.shape

        x = self.proj_in1(x)  # [batch_size, c, h, w]
        attn = self.proj_in2(attn)
        x = rearrange(x, 'b c h w -> b (h w) c')  # [batch_size, h*w, c]
        attn = rearrange(attn, 'b c h w -> b (h w) c')  # [batch_size, h*w, c]

        Q = self.Wq(x)  # [batch_size, h*w, emb_dim]
        K = self.Wk(attn)  # [batch_szie, seq_len, emb_dim]
        V = self.Wv(attn)

        # [batch_size, h*w, seq_len]
        att_weights = torch.einsum('bid,bjd -> bij', Q, K)
        att_weights = att_weights * self.scale

        att_weights = nn.functional.softmax(att_weights, dim=-1)
        out = torch.einsum('bij, bjd -> bid', att_weights, V)  # [batch_size, h*w, emb_dim]

        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)  # [batch_size, c, h, w]
        out = self.proj_out(out)  # [batch_size, c, h, w]

        return out


class Model(nn.Module):
    """
    simply create a 2-branch network, and concat global pooled feature vector.
    each branch = single resnet34
    """
    def __init__(self, type='resnet34', mixstyle_layer=[]):
        super(Model, self).__init__()
        if type == 'resnet50':
            self.backbone_oct = resnet50(pretrained=True)
            self.backbone_fundus = resnet50(pretrained=True)
            expansion = 4
        elif type == 'resnet34':
            self.backbone_oct = resnet34(pretrained=True)
            self.backbone_fundus = resnet34(pretrained=True)
            expansion = 1
        elif type == 'resnet18':
            self.backbone_oct = resnet18(pretrained=True)
            self.backbone_fundus = resnet18(pretrained=True)
            expansion = 1
        else:
            assert NotImplementedError, "No such model type:{}".format(type)

        self.cross_conv1 = CrossConv(64 * expansion, 64 * expansion, 64 * expansion, 2)
        self.cross_conv2 = CrossConv(128 * expansion, 128 * expansion, 128 * expansion, 2)

        self.embedding_layer = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(9, 128),
        )
        self.down = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(512 * expansion * 2, 128), nn.GELU()
        )

        # Head
        self.head = nn.Sequential(
            nn.Dropout(0.1), nn.Linear(128 + 128, 32), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(32, 1)
        )
        self.head_info = nn.Sequential(
            nn.Linear(128, 32), nn.GELU(),
            nn.Linear(32, 1)
        )
        self.head_oct = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(512 * expansion, 128), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(128, 1)
        )
        self.head_fundus = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(512 * expansion, 128), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(128, 1)
        )

        self.sigmoid = nn.Sigmoid()
        # 在oct_branch更改第一个卷积层通道数
        self.backbone_oct.conv1 = nn.Conv2d(256, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mixstyle_layer = mixstyle_layer
        self.TriD = TriD()

    def forward(self, oct, fundus, info, deep_sup=False):
        # Image Features
        oct_fea0 = self.backbone_oct.maxpool(self.backbone_oct.relu(
            self.backbone_oct.bn1(self.backbone_oct.conv1(oct))))
        fundus_fea0 = self.backbone_fundus.maxpool(self.backbone_fundus.relu(
            self.backbone_fundus.bn1(self.backbone_fundus.conv1(fundus))))

        ### Layer1
        oct_fea1 = self.backbone_oct.layer1(oct_fea0)
        fundus_fea1 = self.backbone_fundus.layer1(fundus_fea0)

        if 'layer1' in self.mixstyle_layer:
            oct_fea1 = self.TriD(oct_fea1)
            fundus_fea1 = self.TriD(fundus_fea1)

        ### Information Fusion
        oct_fea1, fundus_fea1 = self.cross_conv1(oct_fea1, fundus_fea1)

        ### Layer2
        oct_fea2 = self.backbone_oct.layer2(oct_fea1)
        fundus_fea2 = self.backbone_fundus.layer2(fundus_fea1)

        if 'layer2' in self.mixstyle_layer:
            oct_fea2 = self.TriD(oct_fea2)
            fundus_fea2 = self.TriD(fundus_fea2)

        ### Information Fusion
        oct_fea2, fundus_fea2 = self.cross_conv2(oct_fea2, fundus_fea2)

        ### Layer3
        oct_fea3 = self.backbone_oct.layer3(oct_fea2)
        fundus_fea3 = self.backbone_fundus.layer3(fundus_fea2)

        ### Layer4
        oct_fea4 = self.backbone_oct.layer4(oct_fea3)
        fundus_fea4 = self.backbone_fundus.layer4(fundus_fea3)

        # Channel-Wise
        oct_fea4 = self.avgpool(oct_fea4).view(oct_fea4.size(0), -1)
        fundus_fea4 = self.avgpool(fundus_fea4).view(fundus_fea4.size(0), -1)

        # Information Embeddings
        one_hot = torch.cat((
            info[:, 1].unsqueeze(-1).float(),
            torch.nn.functional.one_hot(info[:, 0].long(), num_classes=2).float(),
            torch.nn.functional.one_hot(info[:, 2].long(), num_classes=2).float(),
            torch.nn.functional.one_hot(info[:, 3].long(), num_classes=4).float(),
                             ), dim=1)
        info_emb = self.embedding_layer(one_hot).view(oct.size(0), -1)

        # Head
        img_emb = self.down(torch.cat((oct_fea4, fundus_fea4), dim=-1))
        logit = self.head(torch.cat((img_emb, info_emb), dim=-1))

        if deep_sup:
            logit_info = self.head_info(info_emb)
            logit_oct = self.head_oct(oct_fea4)
            logit_fundus = self.head_fundus(fundus_fea4)
            return [process(self.sigmoid(logit), infos=info[:, -1]),
                    process(self.sigmoid(logit_info), infos=info[:, -1]),
                    process(self.sigmoid(logit_oct), infos=info[:, -1]),
                    process(self.sigmoid(logit_fundus), infos=info[:, -1])]
        else:
            return process(self.sigmoid(logit), infos=info[:, -1])


class MixLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, labels, preds):
        r2loss = torch.sum((preds - labels) ** 2) / torch.sum((labels - labels.mean()) ** 2)
        smapeloss = 1 / len(preds) * torch.sum(2 * torch.abs(preds - labels) / (torch.abs(preds) + torch.abs(labels)))
        return smapeloss + r2loss


class R2Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, labels, preds):
        return torch.sum((preds - labels) ** 2) / torch.sum((labels - labels.mean()) ** 2)


class SMAPELoss(nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, labels, preds):
        return 1 / len(preds) * torch.sum(2 * torch.abs(preds - labels) / (torch.abs(preds) + torch.abs(labels)))


def process(logits, infos):
    logits = logits.squeeze()
    label = [3 if i == 0 else i * (-11) for i in infos]
    return logits * torch.tensor(label).to(logits.device)

def Smape_(labels, preds):
    return 1 / len(preds) * torch.sum(2 * torch.abs(preds - labels) / (torch.abs(preds) + torch.abs(labels)))
  
def R2(labels, preds):
    return 1. - torch.sum((preds - labels) ** 2) / torch.sum((labels - labels.mean()) ** 2)

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