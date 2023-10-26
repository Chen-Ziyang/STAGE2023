import torch
import torch.nn as nn
import numpy as np
from resnet import resnet34, resnet50, resnext50_32x4d
from efficientnet_pytorch import EfficientNet


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


class Model(nn.Module):
    """
    simply create a 2-branch network, and concat global pooled feature vector.
    each branch = single resnet34
    """
    def __init__(self):
        super(Model, self).__init__()
        self.backbone = EfficientNet.from_pretrained("efficientnet-b4")

        # 在oct_branch更改第一个卷积层通道数
        self.backbone._conv_stem = nn.Conv2d(256, 48, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone._fc = nn.Sequential()

        self.batch_norm = nn.BatchNorm2d(1792)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.embedding_layer = nn.Linear(9, 256)
        self.linear = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1792 + 256, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 52),
        )

        self.sigmoid = nn.Sigmoid()
        self.activate = LearnableSigmoid()

    def forward(self, input, info):
        one_hot = torch.cat((
            info[:, 1].unsqueeze(-1),
            torch.nn.functional.one_hot(info[:, 0].long(), num_classes=2),
            torch.nn.functional.one_hot(info[:, 2].long(), num_classes=2),
            torch.nn.functional.one_hot(info[:, 3].long(), num_classes=4),
                             ), dim=1)
        info_emb = self.embedding_layer(one_hot).view(input.size(0), -1)
        fea = self.backbone.extract_features(input)

        fea = self.gap(self.batch_norm(fea)).view(input.size(0), -1)
        logit = self.linear(torch.cat((fea, info_emb), dim=-1))
        return logit

# class Model(nn.Module):
#     """
#     simply create a 2-branch network, and concat global pooled feature vector.
#     each branch = single resnet34
#     """
#     def __init__(self, mixstyle_layer=[]):
#         super(Model, self).__init__()
#         self.backbone = resnext50_32x4d(pretrained=True, mixstyle_layer=mixstyle_layer)
#         self.backbone.conv1 = nn.Conv2d(256, 64, kernel_size=7, stride=2, padding=3, bias=False)
#
#         self.embedding_layer = nn.Linear(9, 256)
#         self.linear = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(2048 + 256, 512),
#             nn.GELU(),
#             nn.Dropout(0.1),
#             nn.Linear(512, 52),
#         )
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, input, info):
#         one_hot = torch.cat((
#             info[:, 1].unsqueeze(-1),
#             torch.nn.functional.one_hot(info[:, 0].long(), num_classes=2),
#             torch.nn.functional.one_hot(info[:, 2].long(), num_classes=2),
#             torch.nn.functional.one_hot(info[:, 3].long(), num_classes=4),
#                              ), dim=1)
#         info_emb = self.embedding_layer(one_hot).view(input.size(0), -1)
#         fea = self.backbone(input)
#         logit = self.linear(torch.cat((fea, info_emb), dim=-1))
#         return self.sigmoid(logit) * 40.


class MixLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.smapeloss = SMAPELoss()
        self.mseloss = CustomMSELoss()

    def forward(self, labels, preds, w=10.):
        return self.smapeloss(labels, preds, w) + self.mseloss(labels, preds, w)


class CustomMSELoss(nn.Module):
    def __init__(self):
        super(CustomMSELoss, self).__init__()

    def forward(self, labels, preds, w):
        mask = torch.ones_like(labels)
        mask[labels <= 20] = w
        mask[labels <= 0] = w * 2.
        loss = torch.mean(mask.to(dtype=torch.float32).to(preds.device) * (preds - labels) ** 2)
        return loss


class R2Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, labels, preds, w):
        mask = torch.ones_like(labels)
        mask[labels <= 20] = w
        mask[labels <= 0] = w * 2.
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